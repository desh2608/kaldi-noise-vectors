#!/usr/bin/env bash

set -e -o pipefail

# This script trains an i-vector extractor and then trains an LDA on the 
# noise labels. This is referred to as an e-vector.

stage=0
nj=30
train_set=train_si84   # you might set this to e.g. train.
test_sets=
lda_train_set=
lda_dim=50
ivector_extractor=
lang=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 13 ]; then
  echo "$0: extract features for LDA training data"
  utils/copy_data_dir.sh data/${lda_train_set} data/${lda_train_set}_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/${lda_train_set}_hires
  steps/compute_cmvn_stats.sh data/${lda_train_set}_hires
  utils/fix_data_dir.sh data/${lda_train_set}_hires

  # Get noise labels for training data
  cut -d' ' -f1 data/${lda_train_set}_hires/utt2spk |\
    awk '{print $0,substr($0,length($0),1)}' > data/${lda_train_set}_hires/utt2noise
fi

if [ $stage -le 14 ]; then
  echo "$0: extracting i-vectors for LDA training data"
  nspk=$(wc -l <data/${lda_train_set}_hires/spk2utt)
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj "${nspk}" \
    data/${lda_train_set}_hires ${lang} ${ivector_extractor} \
    exp/nnet3${nnet3_affix}/ivectors_${lda_train_set}_hires
fi 

if [ $stage -le 15 ]; then
  echo "$0: Training LDA"
  ivector-compute-lda --dim=${lda_dim}  --total-covariance-factor=0.1 \
    "ark:ivector-normalize-length scp:exp/nnet3/ivectors_${lda_train_set}_hires/ivectors_utt.scp ark:- |" \
    ark:data/${lda_train_set}_hires/utt2noise \
    ${ivector_extractor}/transform_lda.mat
fi

if [ $stage -le 16 ]; then
  echo "$0: Extracting LDA-based e-vectors for training and test sets"
  for data in $test_sets; do
    num_spk=$(wc -l < "data/${data}_hires/spk2utt")
    nj=$((nj>num_spk ? num_spk : nj))
    local/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
      --transform-mat ${ivector_extractor}/transform_lda.mat \
      data/${data}_hires $lang exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires_lda
  done
fi

exit 0;
