#!/bin/bash
#SBATCH --account=moupe847 
#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --job-name=SentCLS
#SBATCH --output=train-%j.log

conda create -n bert_hw python=3.7
conda activate bert_hw

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install spacy
python -m spacy download en_core_web_sm
pip install nltk
python -m nltk.downloader wordnet


# 6. Run classification task
PREF='cfimdb'
#PREF='sst'
echo "Starting training with prefix: $PREF"

python cls_nlp.py \
    --use_gpu \
    --option finetune \
    --epochs 1 \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "${PREF}-model_task2.pt" 2>&1 | tee "${PREF}-train-log.txt"
    
echo "Ending training with prefix: $PREF"