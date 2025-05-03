### env
```
conda create -n zctr python=3.10.10
conda activate zctr
pip install -r requirements.txt
```

### dataset
1. amazon-2018
- download: https://amazon-reviews-2023.github.io/
```
cd LLaCTR/data/game
gunzip meta_Video_Games.jsonl.gz
gunzip Video_Games.jsonl.gz
python process.py

cd LLaCTR/data/gift
gunzip meta_Gift_Cards.jsonl.gz
gunzip Gift_Cards.jsonl.gz
python process.py

cd LLaCTR/data/magazine
gunzip meta_Magazine_Subscriptions.jsonl.gz
gunzip Magazine_Subscriptions.jsonl.gz
python process.py
```

2. ml
- download: https://grouplens.org/datasets/MovieLens/1m/
```
cd LLaCTR/data/ml1m
unzip ml-1m.zip
python process.py
```

### run backbone
```
cd LLaCTR/ctr

- run
python run_expid.py --config ./config_01 --expid FM_gift --adding_mode 0 --weight_decay 0.01 --gpu 0

- tune
nnictl create --config config_gift.yml -p 8100
```

### run LLaCTR
```
cd LLaCTR/data/gift

python generate_instruction.py

cd LLaCTR/llm

bash train_emb.sh
bash encode.sh
```

