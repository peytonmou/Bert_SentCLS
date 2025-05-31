#!/bin/bash
#SBATCH --account=moupe847 
#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=SentCLS
#SBATCH --output=train-%j.log

module purge
module load anaconda3
module load cuda/11.7

source ~/.bashrc


# Create conda env only if it doesn't exist
if [ ! -d "$HOME/.conda/envs/bert_hw" ]; then
    conda create -n bert_hw python=3.9 -y
fi

conda activate bert_hw

# Install PyTorch compatible with CUDA 11.7 and A100
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==0.24.1  
pip install tokenizers==0.10.1
pip install transformers==4.21.0
pip install datasets==2.0.0 
pip install --upgrade datasets fsspec huggingface_hub

export PATH=$PATH:$HOME/.local/bin

# Run classification task
PREF='sst'
#PREF='cfimdb'

echo "Starting training with prefix: $PREF"

python cls_yelp.py \
    --use_gpu \
    --option pretrain \
    --epochs 1 \
    --lr 5e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "${PREF}-model_yelp_ft.pt" 2>&1 | tee "${PREF}-train-log_yelp_ft.txt"

echo "Ending training with prefix: $PREF"
