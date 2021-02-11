#!/bin/bash

# 1d provides baseline noise embeddings: NAT vector, e-vectors, and bottleneck NNs.

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train_set=train_si84_multi
test_sets="test_eval92 test_0166"
gmm=tri3b        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.

noise_type=bottleneck  # seltzer,evec_lda,bottleneck

# i-vector config (for e-vectors)
num_threads_ubm=8
nj_extractor=10
num_threads_extractor=4
num_processes_extractor=2

nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

affix=1d #affix for TDNN directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/chain${nnet3_affix}/tree_sp

# Stage 1 to 9
local/nnet3/run_chain_common.sh --stage $stage --nj $nj \
  --train-set $train_set --test-sets "$test_sets" \
  --gmm-dir $gmm_dir --ali-dir $ali_dir --lang $lang \
  --lat-dir ${lat_dir} --tree-dir ${tree_dir} \
  --nnet3-affix "$nnet3_affix"

if [ $noise_type == "seltzer" ]; then
  local/nnet3/extract_nat_vectors.sh --stage $stage --nj $nj \
    --train-set ${train_set} --test-sets ${test_sets}
  ivector_affix="_nat"
  ivec_dim=$(feat-to-dim scp:${train_data_dir}/feats.scp - || exit 1;)

elif [ $noise_type == "evec_lda" ]; then
  local/nnet3/run_ivector_common.sh \
    --stage $stage --nj $nj \
    --train-set $train_set --gmm $gmm \
    --test-sets "$test_sets" \
    --num-threads-ubm $num_threads_ubm \
    --nj-extractor $nj_extractor \
    --num-processes-extractor $num_processes_extractor \
    --num-threads-extractor $num_threads_extractor \
    --nnet3-affix "$nnet3_affix"

  local/nnet3/extract_evectors_lda.sh \
    --stage $stage --nj $nj \
    --train-set $train_set \
    --test-sets "$test_sets" \
    --lda-train-set dev_1206 \
    --lda-dim 50 \
    --ivector-extractor exp/nnet3${nnet3_affix}/extractor \
    --lang data/lang

    ivector_affix="_lda"
    ivec_dim=50

elif [ $noise_type == "bottleneck" ]; then
  local/train_noise_dnn.sh data/dev_1206 exp/nnet3_noise_bn
  
  local/nnet3/extract_bottleneck_vectors.sh --nj $nj \
    exp/nnet3_noise_bn data/${train_set}_sp_hires \
    exp/nnet3/ivectors_${train_set}_sp_hires_bottleneck
  
  for test_dir in $test_sets; do
    local/nnet3/extract_bottleneck_vectors.sh --nj 8 \
      exp/nnet3_noise_bn data/${test_dir}_hires \
      exp/nnet3/ivectors_${test_dir}_hires_bottleneck
  done

  ivector_affix="_bottleneck"
  ivec_dim=79
else
  echo "Unknown noise type: ${noise_type}" && exit 1
fi

dir=${dir}${ivector_affix}
train_ivector_dir=${train_ivector_dir}${ivector_affix}

for f in $train_data_dir/feats.scp ${train_ivector_dir}/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 17 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  echo "$0: using noise vectors of dim ${ivec_dim}"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=${ivec_dim} name=ivector
  input dim=40 name=input

  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=1024 input=lda
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 18 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aurora4-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --feat.online-ivector-dir=${train_ivector_dir} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 19 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    data_affix=$(echo $data | sed s/test_//)
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    for lmtype in tgpr_5k; do
      steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context 0 --extra-right-context 0 \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires${ivector_affix} \
        $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1

      cat ${dir}/decode_${lmtype}_${data_affix}/scoring_kaldi/best_wer
    done
  done
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
