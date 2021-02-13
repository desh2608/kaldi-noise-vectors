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
gmm=tri3b                # This specifies a GMM-dir from the features of the type you're training the system on;
lang=
lat_dir=
type=offline
nnet3_affix=

. utils/parse_options.sh

###############################################################################
# Prepare targets for utterances
###############################################################################
segment_dir=exp/segmentation
mkdir -p ${segment_dir}

lang_dir=data/lang
cp ${lang_dir}/phones/silence.txt ${segment_dir}/silence_phones.txt

garbage_phones="OOV"
for p in $garbage_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > ${segment_dir}/garbage_phones.txt

if [ $stage -le 12 ] && [ ! ${segment_dir}/silence_phones.txt ]; then
  echo "$0: Invalid ${segment_dir}/silence_phones.txt"
  exit 1
fi

targets_dir=${segment_dir}/${train_set}_targets
mkdir -p $targets_dir

if [ $stage -le 10 ]; then
  steps/segmentation/lats_to_targets.sh \
    --silence-phones ${segment_dir}/silence_phones.txt \
    --garbage-phones ${segment_dir}/garbage_phones.txt \
    data/${train_set}_sp $lang $lat_dir $targets_dir
fi

if [ $stage -le 11 ]; then
  # Compute speech and noise vectors for training data
  noise_vec_dir=exp/nnet3/noise_${train_set}_sp_hires_offline
  mkdir -p $noise_vec_dir
  
  compute-noise-vector scp:data/${train_set}_sp_hires/feats.scp scp:$targets_dir/targets.scp \
    ark,scp:$noise_vec_dir/noise_vec.ark,$noise_vec_dir/noise_vec.scp
  
  base_feat_dim=$(feat-to-dim scp:data/${train_set}_sp_hires/feats.scp -) || exit 1;
  start_dim=$base_feat_dim
  noise_dim=$((2*base_feat_dim))
  end_dim=$[$base_feat_dim+$noise_dim-1]

  $train_cmd $targets_dir/log/duplicate_feats.log \
    append-vector-to-feats scp:data/${train_set}_sp_hires/feats.scp ark:$noise_vec_dir/noise_vec.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=10 ark:- ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$noise_vec_dir/ivector_online.ark,$noise_vec_dir/ivector_online.scp || exit 1;

  echo 10 > $noise_vec_dir/ivector_period
fi

if [ $stage -le 12 ]; then
  # Segmentation for test data
  utils/mkgraph.sh data/lang_test_tgpr_5k \
    exp/$gmm exp/$gmm/graph_tgpr_5k || exit 1;
  
  for test_dir in ${test_sets}; do
    lang_dir=data/lang_test_tgpr_5k
    targets_dir=${segment_dir}/${test_dir}_targets
    nspk=$(wc -l <data/${test_dir}/spk2utt)

    steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" --skip-scoring true \
      exp/$gmm/graph_tgpr_5k data/${test_dir} exp/$gmm/decode_tgpr_5k_${test_dir} || exit 1;
    
    steps/segmentation/lats_to_targets.sh \
      --acwt 1.0 \
      --frame-subsampling-factor 3 \
      --silence-phones ${segment_dir}/silence_phones.txt \
      --garbage-phones ${segment_dir}/garbage_phones.txt \
      data/${test_dir} $lang exp/$gmm/decode_tgpr_5k_${test_dir} $targets_dir
  done
fi

if [ $stage -le 13 ]; then
  # Compute speech and noise vectors for test data
  for test_dir in $test_sets; do
    targets_dir=${segment_dir}/${test_dir}_targets
    noise_vec_dir=exp/nnet3/noise_${test_dir}_hires_offline
    mkdir -p $noise_vec_dir
    
    compute-noise-vector scp:data/${test_dir}_hires/feats.scp scp:$targets_dir/targets.scp \
      ark,scp:$noise_vec_dir/noise_vec.ark,$noise_vec_dir/noise_vec.scp
    
    base_feat_dim=$(feat-to-dim scp:data/${test_dir}_hires/feats.scp -) || exit 1;
    start_dim=$base_feat_dim
    noise_dim=$((2*base_feat_dim))
    end_dim=$[$base_feat_dim+$noise_dim-1]

    $train_cmd $targets_dir/log/duplicate_feats.log \
      append-vector-to-feats scp:data/${test_dir}_hires/feats.scp ark:$noise_vec_dir/noise_vec.ark ark:- \| \
      select-feats "$start_dim-$end_dim" ark:- ark:- \| \
      subsample-feats --n=10 ark:- ark:- \| \
      copy-feats --compress=true ark:- \
      ark,scp:$noise_vec_dir/ivector_online.ark,$noise_vec_dir/ivector_online.scp || exit 1;

    echo 10 > $noise_vec_dir/ivector_period
  done
fi

exit 0;
