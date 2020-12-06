#!/bin/bash

# evaluate bleu score

# command="$0 $@"
# cmddir=CMDs
# echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
# echo $command >> $cmddir/eval_bleu.cmds

EXP_DIR=/home/dawna/tts/qd212/models/Seq2seq
cd $EXP_DIR

DATA_DIR=/home/dawna/tts/qd212/models/af

# ---------- [model] -------------
model=results/enfr/v0000-tf-nodev-nomask
# libbase=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe
tail=
# tail=.notempty
# ---------- [files] -------------
fname=tst2013 # default seg manual
ftst=${DATA_DIR}/af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr

for i in `seq 1 1 30`
do
    echo $i
    ckpt=$i

    gleuscorer=./local/gleu/compute_gleu.py
    outdir=models/$model/$fname/$ckpt
    srcdir=$ftst.en$tail
    refdir=$ftst.fr$tail

    python $gleuscorer -r $refdir -s $srcdir -o $outdir/translate.txt > $outdir/gleu.log &

done
