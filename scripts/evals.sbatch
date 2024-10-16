#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --qos=gpu-long
#SBATCH --gpus=1
#SBATCH --output=lrz/logs/dump.out
#SBATCH --error=lrz/logs/dump.err
#SBATCH --nodelist=artemis
#SBATCH --job-name=mt_tests_encdec_standard

# Activate your pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv activate ctxeff

python  mt/test.py --model_name transformer_encdec  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-transformer_encdec/fi-en/epoch=04-val_bleu=16.80.ckpt
python  mt/test.py --model_name llama  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --test_per_sample --test_suffix eval --resume_path  data/mt/wmt16-llama/fi-en/epoch=15-val_bleu=13.15.ckpt
python  mt/test.py --model_name retnet  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-retnet/fi-en/epoch=14-val_bleu=12.33.ckpt
python  mt/test.py --model_name mamba  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-mamba/fi-en/epoch=03-val_bleu=13.56.ckpt
python  mt/test.py --model_name mamba_mha  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --layers_config_key reverse_h3 --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-mamba_mha/fi-en/epoch=04-val_bleu=15.35.ckpt
python  mt/test.py --model_name mamba_mistral  --devices 0 --dataset wmt16 --language_pair fi en --use_padding --layers_config_key reverse_h3 --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-mamba_mistral/fi-en/epoch=04-val_bleu=13.74.ckpt --window_size 32
python  mt/test.py --model_name mamba_encdec  --devices 0 --dataset wmt16 --language_pair fi en --use_padding  --test_per_sample --test_suffix eval --resume_path data/mt/wmt16-mamba_encdec/fi-en/epoch=02-val_bleu=15.35.ckpt