#!/bin/bash

set -e -o pipefail

# This scripts uses a trained GMM model to segment the 
# utterances into speech and non-speech frames and estimates
# rough speech and noise vectors by taking the corresponding
# averages.

stage=0
nj=30
train_set=train_si84   # you might set this to e.g. train.
test_sets=

. utils/parse_options.sh

for f in data/${train_set}/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

noise_vec_dir=exp/nnet3/ivectors_${train_set}_sp_hires_nat
mkdir -p $noise_vec_dir

if [ $stage -le 10 ]; then
  echo "$0: Computing NAT vectors for training data"
  compute-noise-vector-seltzer scp:data/${train_set}_sp_hires/feats.scp \
    ark,scp:$noise_vec_dir/noise_vec.ark,$noise_vec_dir/noise_vec.scp
fi

if [ $stage -le 11 ]; then
  base_feat_dim=$(feat-to-dim scp:data/${train_set}_sp_hires/feats.scp -) || exit 1;
  start_dim=$base_feat_dim
  noise_dim=$((base_feat_dim))
  end_dim=$[$base_feat_dim+$noise_dim-1]

  $train_cmd $noise_vec_dir/log/duplicate_feats.log \
    append-vector-to-feats scp:data/${train_set}_sp_hires/feats.scp ark:$noise_vec_dir/noise_vec.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=10 ark:- ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$noise_vec_dir/ivector_online.ark,$noise_vec_dir/ivector_online.scp || exit 1;

  echo 10 > $noise_vec_dir/ivector_period
fi
 
if [ $stage -le 12 ]; then
  # Compute speech and noise vectors for test data
  for test_dir in $test_sets; do
    noise_vec_dir=exp/nnet3/ivectors_${test_dir}_hires_nat
    mkdir -p $noise_vec_dir
    
    compute-noise-vector-seltzer scp:data/${test_dir}_hires/feats.scp \
      ark,scp:$noise_vec_dir/noise_vec.ark,$noise_vec_dir/noise_vec.scp
    
    base_feat_dim=$(feat-to-dim scp:data/${test_dir}_hires/feats.scp -) || exit 1;
    start_dim=$base_feat_dim
    noise_dim=$((base_feat_dim))
    end_dim=$[$base_feat_dim+$noise_dim-1]

    $train_cmd $noise_vec_dir/log/duplicate_feats.log \
      append-vector-to-feats scp:data/${test_dir}_hires/feats.scp ark:$noise_vec_dir/noise_vec.ark ark:- \| \
      select-feats "$start_dim-$end_dim" ark:- ark:- \| \
      subsample-feats --n=10 ark:- ark:- \| \
      copy-feats --compress=true ark:- \
      ark,scp:$noise_vec_dir/ivector_online.ark,$noise_vec_dir/ivector_online.scp || exit 1;

    echo 10 > $noise_vec_dir/ivector_period
  done
fi

exit 0;
