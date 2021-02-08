#!/usr/bin/env bash

set -e -o pipefail

stage=0
nj=30

train_set=train_si84_multi
test_sets="eval92 0166"
gmm_dir=tri3b     # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
ali_dir=
lat_dir=
lang=
tree_dir=

nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

# Multi-condition config
norvb_datadir=data/${train_set}_sp
rvb_affix=_rvb
num_data_reps=1
sample_rate=16000
max_jobs_run=10

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

# low-resolution features and alignments,
if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 2 ]; then
  echo "$0: data/${train_set}_sp/feats.scp already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data (needed for alignments)"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 3 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir
fi

# high-resolution features and i-vector extractor,
if [ $stage -le 4 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: data/${train_set}_sp_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 4."
  exit 1
fi

if [ $stage -le 4 ]; then
  echo "$0: creating high-resolution MFCC features"

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 5 ]; then
  echo "$0: creating reverberated MFCC features"

  if [ ! -f ${norvb_datadir}${rvb_affix}${num_data_reps}_hires/feats.scp ]; then
    if [ ! -d "RIRS_NOISES/" ]; then
      # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
      wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
      unzip rirs_noises.zip
    fi

    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

    python steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs "20:10:15:5:0" \
      --background-snrs "20:10:15:5:0" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications ${num_data_reps} \
      --max-noises-per-minute 1 \
      --source-sampling-rate $sample_rate \
      ${norvb_datadir} ${norvb_datadir}${rvb_affix}${num_data_reps}

    utils/copy_data_dir.sh ${norvb_datadir}${rvb_affix}${num_data_reps} ${norvb_datadir}${rvb_affix}${num_data_reps}_hires
    utils/data/perturb_data_dir_volume.sh ${norvb_datadir}${rvb_affix}${num_data_reps}_hires

    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd --max-jobs-run $max_jobs_run" ${norvb_datadir}${rvb_affix}${num_data_reps}_hires
    steps/compute_cmvn_stats.sh ${norvb_datadir}${rvb_affix}${num_data_reps}_hires
    utils/fix_data_dir.sh ${norvb_datadir}${rvb_affix}${num_data_reps}_hires
  fi

  utils/combine_data.sh data/${train_set}_comb_hires data/${train_set}_sp_hires ${norvb_datadir}${rvb_affix}${num_data_reps}_hires
fi

if [ $stage -le 6 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj ${nj} --cmd "$train_cmd" data/${train_set}_sp \
    $lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space

  # Copy lattices for reverberated copies
  local/reverberate_lat_dir.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_comb_hires ${lat_dir} ${lat_dir}_comb
fi

if [ $stage -le 8 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (stage 3), so use those.  The num-leaves is always somewhat less 
  # than the num-leaves from the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 data/${train_set}_sp \
    $lang $ali_dir $tree_dir
fi

if [ $stage -le 9 ]; then
  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr_5k/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgpr_5k \
    $tree_dir $tree_dir/graph_tgpr_5k || exit 1;

fi

exit 0
