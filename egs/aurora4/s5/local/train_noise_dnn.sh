#!/usr/bin/env bash

# Train a DNN to predict noise labels from frames.
# Bottleneck layer will be used to extract noise embedding.

. ./cmd.sh

affix=
stage=0
train_stage=-10
get_egs_stage=-10
egs_opts=

remove_egs=true
nj=40

chunk_width=20

# The context is chosen to be around 1 second long. The context at test time
# is expected to be around the same.
extra_left_context=79
extra_right_context=21

relu_dim=512
bottleneck_dim=80

# training options
num_epochs=10
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=6
num_jobs_final=12
max_param_change=0.2  # Small max-param change for small network


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

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <data-dir> <dir>\n" `basename $0`
  exit 1;
fi

data_dir=$1
dir=$2

mkdir -p $dir

if [ ! -f ${data_dir}_hires/feats.scp ]; then
  echo "$0: extract features for training data"
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" ${data_dir}_hires
  steps/compute_cmvn_stats.sh ${data_dir}_hires
  utils/fix_data_dir.sh ${data_dir}_hires
fi

if [ $stage -le 0 ]; then
  echo "$0: getting targets"
  # Get noise labels for training data
  cut -d' ' -f1 ${data_dir}_hires/utt2spk |\
    awk '{print $0,substr($0,length($0),1)}' > ${data_dir}_hires/utt2noise
  
  local/get_noise_targets.py ${data_dir}_hires/utt2noise ${data_dir}/utt2num_frames - |\
    copy-feats ark,t:- ark,scp:$dir/targets.ark,$dir/targets.scp
fi

samples_per_iter=`perl -e "print int(400000 / $chunk_width)"`
cmvn_opts="--norm-means=false --norm-vars=false"
echo $cmvn_opts > $dir/cmvn_opts

if [ $stage -le 1 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=`feat-to-dim scp:${data_dir}_hires/feats.scp -` name=input
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat 
  
  relu-renorm-layer name=tdnn1 input=lda dim=$relu_dim add-log-stddev=true
  relu-renorm-layer name=tdnn2 dim=$relu_dim add-log-stddev=true
  relu-renorm-layer name=bn dim=$bottleneck_dim add-log-stddev=true
  relu-renorm-layer name=tdnn3 dim=$relu_dim add-log-stddev=true
  output-layer name=output include-log-softmax=true dim=14 learning-rate-factor=0.1
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/

  cat <<EOF >> $dir/configs/vars
num_targets=14
EOF
  
  echo "output-node name=output input=bn.affine" > $dir/extract.config
fi

if [ $stage -le 2 ]; then
  num_utts=`cat $data_dir/utt2spk | wc -l`
  # Set num_utts_subset for diagnostics to a reasonable value
  # of max(min(0.005 * num_utts, 300), 12)
  num_utts_subset=`perl -e '$n=int($ARGV[0] * 0.005); print ($n > 300 ? 300 : ($n < 12 ? 12 : $n))' $num_utts`

  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="$cmvn_opts" \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage \
    --egs.chunk-left-context=$extra_left_context \
    --egs.chunk-right-context=$extra_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.rnn.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.5 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.max-param-change=$max_param_change \
    --trainer.compute-per-dim-accuracy=true \
    --cmd="$train_cmd" --nj $nj \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --use-dense-targets=true \
    --feat-dir=${data_dir}_hires \
    --targets-scp="$dir/targets.scp" \
    --egs.opts="--frame-subsampling-factor 1 --num-utts-subset $num_utts_subset" \
    --dir=$dir || exit 1
fi
