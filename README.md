# NCVPRIPG 2023 Challenge on Writer Verification (Finalist Solution)

![intro image](assets/img.png)

**Challenge Webpage**: [Summer Challenge on Writer Verification, under NCVPRIPG'23](https://vl2g.github.io/challenges/wv2023)  
**Web Link**: [Winning the NCVPRIPG23 Challenge on Writer Verification](https://mohitsharma-iitj.github.io/ncvpripg23_writer_verification/)  
**Kaggle Link**: [Summer Challenge on Writer Verification](https://www.kaggle.com/competitions/summer-challenge-on-writer-verification23-finale/leaderboard)  
**Team Name**: Alpha  
**Summary Paper**: `assets/NCVPRIPG23_Writer_Verification.pdf`


## Environment

```shell
python3 -m pip install requirement.txt
pip install gdown
# to upgrade
pip install --upgrade gdown
mkdir -p data/raw
```

Download and extract the dataset including the val and test CSVs in `data/raw`.

## Download Checkpoints
The checkpoint for the best trained model can be found at [Google Drive](https://drive.google.com/drive/folders/1AAAuh62G2LHKOPOJvhUkUuC7w6SMKvzI?usp=sharing). You can download this and store it at a desired location. For downloading, you can use [gdown](https://github.com/wkentaro/gdown).

```shell
mkdir pretrained
cd pretrained
gdown 'https://drive.google.com/uc?id=1848Iu-JKXWSBgFvBN50-l-ZXXKgqkRJf'
```

## Generating Results on Test Set

```shell
python inference.py --ckpt pretrained/model_best_base.pth.tar
```

## Training

To train the model again, extract the base code and run code below.

To start training, run
```shell
python train.py
```

To reproduce the best results, pass the best config
```shell
python train.py --config cfgs/best.yml
```

To disable wandb while training, run
```shell
WANDB_MODE=disabled python train.py
```
