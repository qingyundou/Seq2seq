#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD # overwrite env
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

AIR_FORCE_GPU=0
export MANU_CUDA_DEVICE=0 #note on nausicaa no.2 is no.0
# select gpu when not on air
if [[ "$HOSTNAME" != *"air"* ]]  || [ $AIR_FORCE_GPU -eq 1 ]; then
  X_SGE_CUDA_DEVICE=$MANU_CUDA_DEVICE
fi
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo "on $HOSTNAME, using gpu (no nb means cpu) $CUDA_VISIBLE_DEVICES"

# activate your conda env
# python 3.6
# pytorch 1.1/1.3
# source activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# qd212
source /home/mifs/ytl28/anaconda3/etc/profile.d/conda.sh
# conda activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# conda activate pt12-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt12-cuda10/bin/python3
conda activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# conda activate pt15-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt15-cuda10/bin/python3

# ----- dir ------
EXP_DIR=/home/dawna/tts/qd212/models/Seq2seq
cd $EXP_DIR

DATA_DIR=/home/dawna/tts/qd212/models/af

path_vocab_src=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en
path_vocab_tgt=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr
use_type='word'

# libbase=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe

# ------ [orig] ------

# fname=tst2013
# ftst=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en
fname=tst2014
ftst=${DATA_DIR}/af-lib/en-fr-2015/iwslt15_en_fr/IWSLT16.TED.tst2014.en-fr.en
seqlen=200

# ------ [after dd] ------

# fname=test_dtal_afterdd
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-dtal/dtal.flt
# seqlen=165


# ----- models ------
# export ckpt=$1
model=results/enfr/v0000-tf-nodev-nomask
# ckpt=18
tmp=checkpoints_epoch
# ckpt=2020_12_03_00_25_03 # 2020_12_03_02_04_53
# tmp=checkpoints
beam_width=1
batch_size=50
use_gpu=True




for f in ${EXP_DIR}/$model/checkpoints_epoch/*; do
    ckpt=$(basename $f)
    test_path_out=$model/$fname/$ckpt/
    if [ ! -f "${test_path_out}translate.txt" ]; then
    echo MODE: translate, save to $test_path_out

    $PYTHONBIN ${EXP_DIR}/translate.py \
        --test_path_src $ftst \
        --seqrev False \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --use_type $use_type \
        --load $model/$tmp/$ckpt \
        --test_path_out $test_path_out \
        --max_seq_len $seqlen \
        --batch_size $batch_size \
        --use_gpu $use_gpu \
        --beam_width $beam_width \
        --eval_mode 1
    fi
done