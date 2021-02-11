#!/usr/bin/env bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
#               2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.

# This script extracts noise embeddings from a bottleneck NN. It is based on
# the x-vector extraction scripts.

# Begin configuration section.
nj=30
cmd="run.pl"

cache_capacity=64 # Cache capacity for x-vector extractor
chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet
                  # directory.
use_gpu=false
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <out-dir>"
  echo " e.g.: $0 exp/nnet_noise data/dev exp/ivectors_dev_bottleneck"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --cache-capacity <n|64>                          # To speed-up xvector extraction"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.raw $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

echo "$0: extracting bottleneck noise vectors for $data"

###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################

frame_subsampling_factor=1

if [ $stage -le 1 ]; then

  ########################################################################
  ## Initialize neural network for decoding using the output $output_name
  ########################################################################
  iter=final
  if [ -f $srcdir/extract.config ] ; then
    $cmd $dir/log/get_nnet_final.log \
      nnet3-copy --nnet-config=$srcdir/extract.config $srcdir/final.raw \
      $srcdir/final_bn.raw || exit 1
    iter=final_bn
  fi

  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter ${iter} \
    --extra-left-context 79 \
    --extra-right-context 21 \
    --frame-subsampling-factor $frame_subsampling_factor \
    $data $srcdir $dir || exit 1
fi

mv $dir/output.scp $dir/ivector_online.scp
echo 1 > $dir/ivector_period

exit 0
