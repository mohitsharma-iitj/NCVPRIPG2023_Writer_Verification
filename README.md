# NCVPRIPG 2023 Challenge on Writer Verification (Finalist Solution)

![intro image](assets/img.png)

**Challenge Webpage**: [Summer Challenge on Writer Verification, under NCVPRIPG'23](https://vl2g.github.io/challenges/wv2023)  
**Web Link**: [Finalist of NCVPRIPG23 Challenge on Writer Verification](https://mohitsharma-iitj.github.io/NCVPRIPG2023_Writer_Verification/)  
**Kaggle Link**: [Summer Challenge on Writer Verification](https://www.kaggle.com/competitions/summer-challenge-on-writer-verification23-finale/leaderboard)  
**Team Name**: Alpha  
**Summary Paper**: `assets/NCVPRIPG23_Writer_Verification.pdf`

**Python Version**: `3.10.5`

**Tensorflow Version**: `2.13`



## Environment
Create the required directories by download manually from [Google Drive](https://drive.google.com/drive/folders/1xhPeBt5VeRWNnY8SKLu-cyLjm8CjjdQD?usp=sharing) extract it and open 'notebook' in vscode terminal or command prompt like cd c:\..path..\notebook . You will see requirement.txt, test_pretrained_model.py etc in it. You have path like C:\Users\......\notebook-20230804T162632Z-001\notebook>. Then create the virtual environment in notebook folder as ->

```shell
python -m venv virtual_env
cd virtual_env\Scripts
.\activate.bat
cd ../..
```
NOTE: You can also clone this repo to evaluate the result. Do whatever you like, but give path correct as below code run in notebook folder.


## Libraries
Install the required libraries. {give your own directory path}
```shell
pip install -r requirenment.txt
```



## Download Checkpoints (if training model again)
If you want to train the model again then download this and store it at a desired location. Initially only few images are there to reduce download size. Download files should be saved at `notebook/train`,`notebook/val`,`notebook/test` after extracting from this [Google Drive link](https://drive.google.com/drive/folders/1kQG-b9Jvha05d-5xQ2yK7cI5pVRSXzex?usp=sharing).



## Generating Results on Test Set
On pretrained model (take input from notebook/test).
NOTE: to check result on custom input save 2 image in notebook/test as img1.jpg , img2.jpg.
```shell
python test_pretrained_model.py 
```
On self-trained model (trained on -notebook/train, notebook/val && test on - notebook/test).
Have to train it first.
```shell
python train.py 
python test.py 
```




# Thank You
