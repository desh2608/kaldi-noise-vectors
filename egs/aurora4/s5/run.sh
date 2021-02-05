#!/usr/bin/env bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

stage=0
train_set=multi # Set this to 'clean' or 'multi'
test_sets="eval92 0166"

. utils/parse_options.sh

aurora4=/export/corpora5/AURORA

#we need lm, trans, from WSJ0 CORPUS
wsj0=/export/corpora5/LDC/LDC93S6B

if [ $stage -le 0 ]; then
  local/aurora4_data_prep.sh $aurora4 $wsj0
fi

if [ $stage -le 1 ]; then
  local/wsj_prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang
fi

if [ $stage -le 2 ]; then
  local/aurora4_format_data.sh
fi

mfccdir=mfcc
if [ $stage -le 3 ]; then
  # Now make MFCC features.
  for x in train_si84_multi test_eval92 test_0166 dev_1206; do 
   steps/make_mfcc.sh  --nj 10 \
     data/$x exp/make_mfcc/$x $mfccdir || exit 1;
   steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  # mono
  steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    data/train_si84_multi data/lang exp/mono0a || exit 1;
fi

if [ $stage -le 5 ]; then
  # tri1
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/train_si84_multi data/lang exp/mono0a exp/mono0a_ali || exit 1;

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train_si84_multi data/lang exp/mono0a_ali exp/tri1 || exit 1;
fi

if [ $stage -le 6 ]; then
  # tri2
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train_si84_multi data/lang exp/tri1 exp/tri1_ali_si84 || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 \
    data/train_si84_multi data/lang exp/tri1_ali_si84 exp/tri2a || exit 1;

  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train_si84_multi data/lang exp/tri2a exp/tri2a_ali_si84 || exit 1;
  
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      2500 15000 data/train_si84_multi data/lang exp/tri2a_ali_si84 exp/tri2b || exit 1;
fi

if [ $stage -le 7 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.
  # Align tri2b system with all the si84 data.
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
    data/train_si84_multi data/lang exp/tri2b exp/tri2b_ali_si84  || exit 1;
  
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_si84_multi data/lang exp/tri2b_ali_si84 exp/tri3b || exit 1;
fi

exit 0;
