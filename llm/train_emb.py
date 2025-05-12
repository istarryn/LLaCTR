import os
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
# from accelerate import Accelerator

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

import pdb
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

def train(
    # model/data params
    base_model: str = "./Meta-Llama-3-8B",  # the only required argument
    train_data_path: List[str] = ["./data/box/train.json"],
    val_data_path: List[str] = ["./data/box/valid.json"],
    output_dir: str = "./model/test",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # accelerator = Accelerator()
    transformers.set_seed(seed)
    ############################# ddp ##############################
    device_map = "auto" # 
    print("device_map =", device_map)

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    ############################# load model ##############################
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, 
    #     bnb_4bit_use_double_quant=True, 
    #     bnb_4bit_quant_type="nf4", 
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    if resume_from_checkpoint != "":
        ############################# load old lora ##############################
        config = PeftConfig.from_pretrained(
            resume_from_checkpoint
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_cache = False
        )
        model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
        model.print_trainable_parameters()
    else:
        ############################# create new lora ##############################
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_cache = False
        )
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    ############################# load tokenizer ##############################
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    print("len(tokenizer) =", len(tokenizer))
    print(f"tokenizer.eos_token_id = {tokenizer.eos_token_id}; {tokenizer.decode(tokenizer.eos_token_id)}")
    print(f"tokenizer.pad_token_id = {tokenizer.pad_token_id}; {tokenizer.decode(tokenizer.pad_token_id)}")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    ############################# prepare data ##############################
    train_data_list = []
    val_data_list = []

    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))
    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    def generate_prompt(data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        # only train on outputs 
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])       
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]           
        
        if tokenized_full_prompt["labels"][-2] == -100:
            print("EORROR: input_ids > cutoff_len!")
            print("full_prompt:", full_prompt)
        return tokenized_full_prompt

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))

    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])

    ############################# start training ##############################
    model.enable_input_require_grads()
    ### amazon
    field_prompt_list = [
        'parent_asin',
        'user_id',
        'timestamp',
        'main_category',
        'title',
        'average_rating',
        'rating_number',
        'features',
        'description',
        'price',
        'store',
        'categories',
        'details',
    ]      

    ### ml1m
    # field_prompt_list = [
    #     'UserID',
    #     'Gender',
    #     'Age',
    #     'Occupation',
    #     'MovieID',
    #     'Title',
    #     'Genres',
    #     'Timestamp',
    # ]       


    input_field = tokenizer(field_prompt_list, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    
    input_field_id = input_field.input_ids
    input_field_attention_mask = input_field.attention_mask

    # DEBUG
    # tokenizer = AutoTokenizer.from_pretrained('./Llama3_Checkpoints')
    # tokenizer('### Response:\n', add_special_tokens=False)
    # tokenizer.decode([14711, 6075, 512])
    def find_512_indices_from_end(query_input_ids):
        B, L = query_input_ids.size()
        return L - torch.argmax((torch.flip(query_input_ids, dims=[1]) == 512).int(), dim=1) - 1

    field_result = []
    for i in range(input_field_id.size(0)):
        field_ids = input_field_id[i]  # (L,)
        attention_mask = input_field_attention_mask[i]  # (L,)
        masked_ids = field_ids[attention_mask == 1]
        field_result.append(masked_ids)

    class MyTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):                     
            query_input_ids = inputs["input_ids"] # [B, L]
            query_attention_mask = inputs["attention_mask"]
            labels_input_ids = inputs["labels"]           
            model_output = model(input_ids=query_input_ids, attention_mask=query_attention_mask, output_hidden_states=True)
            last_hidden_states = model_output.hidden_states[-1] # [B, L, D]  
            # get query emb
            batch_idx = torch.arange(last_hidden_states.size(0))  
            query_emb_eos_indices = find_512_indices_from_end(query_input_ids)            
            query_emb_eos = last_hidden_states[batch_idx, query_emb_eos_indices, :] # [B, D]
            query_emb_eos_norm = query_emb_eos / torch.norm(query_emb_eos, p=2, dim=-1, keepdim=True)   
            
            # get all fields emb
            outputs_field = model(input_ids=input_field_id, attention_mask=input_field_attention_mask, output_hidden_states=True)
            field_emb_eos = outputs_field.hidden_states[-1][:, -1, :] # [NUM_FIELD, D]
            field_emb_eos_norm = field_emb_eos / torch.norm(field_emb_eos, p=2, dim=-1, keepdim=True)
            

            # get answer field id
            batch_answer_field_id = []
            for i_b in range(query_input_ids.size(0)):
                begin_index = query_emb_eos_indices[i_b].item()+1
                answer_field_id = query_input_ids[i_b][begin_index:-1]
                batch_answer_field_id.append(answer_field_id)
            # import pdb
            # pdb.set_trace()                 
            result_indices = torch.zeros(len(batch_answer_field_id), dtype=torch.long, device=field_emb_eos_norm.device)
            for i, answer_field in enumerate(batch_answer_field_id):
                found = False
                for j, field in enumerate(field_result):
                    if torch.equal(answer_field, field):
                        result_indices[i] = j
                        found = True
                        break
                if not found:
                    print(f"Element {answer_field} not found in field_result")
                    exit(0)
            labels_emb_eos_norm = field_emb_eos_norm[result_indices] # [B, D] 

            # infonce loss           
            tau = 0.02
            sim_metric = torch.matmul(query_emb_eos_norm, labels_emb_eos_norm.transpose(0, 1)) # BXB
            pos_score = torch.exp(torch.diag(sim_metric) / tau) # BX1
            # neg space is other fields           
            my_neg = torch.matmul(query_emb_eos_norm, field_emb_eos_norm.transpose(0, 1)) # BX10
            neg_score = torch.sum(torch.exp(my_neg / tau), dim=-1) # BX1
            loss = -torch.mean(torch.log(pos_score / neg_score))            
            return (loss, model_output) if return_outputs else loss

    trainer = MyTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            # per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True, # fp bf
            logging_steps=10,
            optim="adamw_torch",
            # evaluation_strategy="epoch", # evaluation_strategy="steps", eval_steps=200 
            save_strategy="epoch", # save_strategy="steps", save_steps=200
            output_dir=output_dir,
            # save_total_limit=1,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard", # report_to=None,
            save_on_each_node=False,
            overwrite_output_dir=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

if __name__ == "__main__":
    fire.Fire(train)









