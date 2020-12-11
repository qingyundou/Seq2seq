#!/bin/bash
#$ -S /bin/bash

# ------------------------ ENV --------------------------
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
conda activate pt11-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# conda activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# conda activate pt15-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt15-cuda10/bin/python3

# conda list | grep tensorboard
# tensorboard --logdir=runs

# ------------------------ DIR --------------------------
EXP_DIR=/home/dawna/tts/qd212/models/Seq2seq
cd $EXP_DIR

DATA_DIR=/home/dawna/tts/qd212/models/af

savedir=results/enfr/v0000-tf-nodev-nomask
train_path_src=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.en
train_path_tgt=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.fr
# dev_path_src=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en
# dev_path_tgt=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.fr
dev_path_src=None
dev_path_tgt=None
path_vocab_src=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en
path_vocab_tgt=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr
use_type='word'
load_embedding_src=None
load_embedding_tgt=None
share_embedder=False

# ------------------------ MODEL --------------------------
embedding_size_enc=200
embedding_size_dec=200
hidden_size_enc=200
hidden_size_dec=200
hidden_size_shared=200
num_bilstm_enc=2
num_unilstm_dec=4
att_mode=bilinear # bahdanau | bilinear

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=6000
print_every=1000

batch_size=50 # 256
max_seq_len=64 # 32
minibatch_split=1
num_epochs=60 # 20

random_seed=2020
eval_with_mask=False
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True

load=$savedir/checkpoints_epoch/34

$PYTHONBIN ${EXP_DIR}/train.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--path_vocab_src $path_vocab_src \
	--path_vocab_tgt $path_vocab_tgt \
	--load_embedding_src $load_embedding_src \
	--load_embedding_tgt $load_embedding_tgt \
	--use_type $use_type \
	--save $savedir \
	--random_seed $random_seed \
	--share_embedder $share_embedder \
	--embedding_size_enc $embedding_size_enc \
	--embedding_size_dec $embedding_size_dec \
	--hidden_size_enc $hidden_size_enc \
	--num_bilstm_enc $num_bilstm_enc \
	--num_unilstm_enc 0 \
	--hidden_size_dec $hidden_size_dec \
	--num_unilstm_dec $num_unilstm_dec \
	--hidden_size_att 10 \
	--att_mode $att_mode \
	--residual True \
	--hidden_size_shared $hidden_size_shared \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--seqrev False \
	--eval_with_mask $eval_with_mask \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--learning_rate 0.001 \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
	--normalise_loss $normalise_loss \
	--minibatch_split $minibatch_split \
	--load $load