#!/usr/bin/env bash

set -e -o pipefail

# This script is called from scripts like local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more scripts).  It
# contains the common feature preparation and iVector-related parts of the
# script.  See those scripts for examples of usage.

stage=0
nj=30
train_set=train_si84   # you might set this to e.g. train.
test_sets=
gmm=tri4b                # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.
online_cmvn_iextractor=false

num_threads_ubm=32

nj_extractor=10
# It runs a JOB with '-pe smp N', where N=$[threads*processes]
num_processes_extractor=4
num_threads_extractor=4

nnet3_affix=             # affix for exp/nnet3 directory to put iVector stuff in (e.g.
                         # in the tedlium recip it's _cleaned).
ivector_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

extract_ivectors="steps/online/nnet2/extract_ivectors.sh"
if [ -z ${ivector_affix} ]; then
  lang_opts="data/lang"
  extract_ivectors="local/extract_ivectors.sh"
fi

if [ $stage -le 10 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
      $num_utts ${temp_data_root}/${train_set}_sp_hires_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_sp_hires_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm

fi

if [ $stage -le 11 ]; then
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" \
    --online-cmvn-iextractor $online_cmvn_iextractor \
    --nj $nj_extractor --num-threads $num_threads_extractor --num-processes $num_processes_extractor \
    data/${train_set}_sp_hires exp/nnet3${nnet3_affix}/diag_ubm exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 12 ]; then
  
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires${ivector_affix}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/aurora4-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi

  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2

  steps/online/nnet2/extract_ivectors${ivector_affix}.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_max2 ${lang_opts} \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  for data in ${test_sets}; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    "${extract_ivectors}" --cmd "$train_cmd" --nj "${nspk}" \
      data/${data}_hires ${lang_opts} exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires${ivector_affix}
  done
fi

exit 0;
