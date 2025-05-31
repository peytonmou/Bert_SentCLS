# BERT Sentiment Classification 
This project implements and fine-tunes a BERT model for sentiment classification on **SST-5** and **IMDB** datasets. It also explores several experimental methods to enhance performance.

## Environment Setup
Tested with Python 3.7 and CUDA 10.1
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.58.0 requests==2.25.1 importlib-metadata==3.7.0 filelock==3.0.12
pip install sklearn==0.0 tokenizers==0.10.1 transformers==4.21.0 datasets==2.0.0

## Task 1: Model Replication
Re-implemented core BERT components from scratch, achieved baseline dev accuracy:
- IMDB: 97.1%
- SST-5: 52.4%

## Task 2: Experiments for Performance Improvements
- Hyperparameter tuning
- POS tagging, dependency parsing, WordNet
- Custom loss function
- External dataset pretraining  
Results: Minor improvement only — SST-5 test accuracy increased to 53.4% (+1.5%)

## Repository Structue:
- code/          base_bert, bert, classifier, tokenizer, optimizer, config, utils
- data/          SST-5 and IMDB dataset splits
- logs/          training & testing logs
- output/        dev & test outputs
- report.pdf     explaining model architecture and experiments
- run.sh         environment setup + run script
