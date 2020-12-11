#!/bin/bash
#$ -S /bin/bash


# ------------------------ MODE --------------------------
MODE=train # train translate translate_attention


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
# conda activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
conda activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# conda activate pt15-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt15-cuda10/bin/python3

# conda list | grep tensorboard
# tensorboard --logdir=runs

# if [[ "$MODE" == "train" ]]; then
# 	conda activate pt15-cuda10
# 	export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt15-cuda10/bin/python3
# else
# 	conda activate py13-cuda9
# 	export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# fi
# case $MODE in
# "train")
# 	conda activate pt15-cuda10
# 	export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt15-cuda10/bin/python3
# ;;
# "translate")
# 	conda activate py13-cuda9
# 	export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# ;;
# esac


# ------------------------ DIR --------------------------
EXP_DIR=/home/dawna/tts/qd212/models/Seq2seq
cd $EXP_DIR

DATA_DIR=/home/dawna/tts/qd212/models/af

# savedir=results/enfr/v0010-mp-p2
savedir=results/enfr/v0010-mp-p2-stable

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
num_epochs=50 # 20

random_seed=2020
eval_with_mask=False
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True

# init_dir=results/enfr/v0000-tf-nodev-nomask/checkpoints_epoch/30
# init_dir_p2=$init_dir # None
init_dir=results/enfr/v0000-tf-nodev-nomask/checkpoints_epoch/25
init_dir_p2=results/enfr/v0010-mp-p2/checkpoints_epoch/31
# load_p2=results/enfr/v0010-mp-p2/checkpoints_epoch/30

# ------------------------ TEST --------------------------
# ----- data ------
# fname=tst2013
# ftst=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en
fname=tst2014
ftst=${DATA_DIR}/af-lib/en-fr-2015/iwslt15_en_fr/IWSLT16.TED.tst2014.en-fr.en
seqlen=200

# ----- models ------
# export ckpt=$1
model=$savedir
ckpt=31 # 22
tmp=checkpoints_epoch
# ckpt=2020_12_03_00_25_03 # 2020_12_03_02_04_53
# tmp=checkpoints
beam_width=1
batch_size=50
use_gpu=True

# ------------------------ RUN --------------------------
echo MODE: $MODE
case $MODE in
"train")
	$PYTHONBIN ${EXP_DIR}/train-mp-p2.py \
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
		--init_dir $init_dir \
		--init_dir_p2 $init_dir_p2 \
		# --load $init_dir \
		# --load_p2 $load_p2
;;
"translate")
trap "exit" INT
for f in ${EXP_DIR}/$model/checkpoints_epoch/*; do
    ckpt=$(basename $f)
    test_path_out=$model/$fname/$ckpt/
    if [ ! -f "${test_path_out}translate.txt" ]; then
    echo MODE: translate, save to $test_path_out
    $PYTHONBIN ${EXP_DIR}/translate-mp-p2.py \
        --test_path_src $ftst \
        --seqrev False \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --use_type $use_type \
        --load $init_dir \
        --load_p2 $model/$tmp/$ckpt \
        --test_path_out $test_path_out \
        --max_seq_len $seqlen \
        --batch_size $batch_size \
        --use_gpu $use_gpu \
        --beam_width $beam_width \
        --eval_mode 1
    fi
done
;;
"translate_attention")
	ckpt=$ckpt
	test_path_out=$model/$fname/$ckpt/
	echo MODE: translate_attention, save to $test_path_out
	$PYTHONBIN ${EXP_DIR}/translate-mp-p2.py \
	    --test_path_src $ftst \
	    --seqrev False \
	    --path_vocab_src $path_vocab_src \
	    --path_vocab_tgt $path_vocab_tgt \
	    --use_type $use_type \
	    --load $init_dir \
	    --load_p2 $model/$tmp/$ckpt \
	    --test_path_out $test_path_out \
	    --max_seq_len $seqlen \
	    --batch_size $batch_size \
	    --use_gpu $use_gpu \
	    --beam_width $beam_width \
	    --eval_mode 3
;;
esac