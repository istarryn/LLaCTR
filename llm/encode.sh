# bash encode.sh

for lora_weights in "model/ml1m/sl_1000/checkpoint-32" "model/ml1m/sl_1000/checkpoint-65" "model/ml1m/sl_1000/checkpoint-98" "model/ml1m/sl_1000/checkpoint-131" "model/ml1m/sl_1000/checkpoint-160"
do
    python encode.py \
        --prompt_path "llm_emb/ml1m/field.txt" \
        --llm_emb_path "${lora_weights}/field_sl_eos.pt" \
        --lora_weights $lora_weights \
        --base_model "/c23034/wbh/Llama3_Checkpoints" 
done