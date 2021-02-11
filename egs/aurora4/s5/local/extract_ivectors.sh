#!/usr/bin/env bash

# Copyright     2013  Daniel Povey
# Apache 2.0.


# This is the same as steps/online/nnet2/extract_ivectors.sh, except that there
# is an additional option for providing a transform_mat by which the extracted
# i-vectors are transformed. Also, the sub-speaker-frames option is removed since
# this script only extracts per utterance i-vectors.


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
ivector_period=10
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.
max_count=100       # Interpret this as a number of frames times posterior scale...
                    # this config ensures that once the count exceeds this (i.e.
                    # 1000 frames, or 10 seconds, by default), we start to scale
                    # down the stats, accentuating the prior term.   This seems quite
                    # important for some reason.

compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).
silence_weight=0.0
acwt=0.1  # used if input is a decode dir, to get best path from lattices.
mdl=final  # change this if decode directory did not have ../final.mdl present.
num_threads=1 # Number of threads used by ivector-extract.  It is usually not
              # helpful to set this to > 1.  It is only useful if you have
              # fewer speakers than the number of jobs you want to run.
transform_mat= # If provided, ivectors will be transformed using this LDA transform matrix
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 [options] <data> <lang> <extractor-dir> <ivector-dir>"
  echo " e.g.: $0 data/test data/lang exp/nnet2_online/extractor exp/tri3/decode_test exp/nnet2_online/ivectors_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-threads)"
  echo "  --num-threads <n|1>                              # Number of threads for each job"
  echo "                                                   # Ignored if <alignment-dir> or <decode-dir> supplied."
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  echo "  --posterior-scale <float;default=0.1>            # Scale on posteriors in iVector extraction; "
  echo "                                                   # affects strength of prior term."
  echo "  --transform-mat <file>                           # LDA transformation matrix file."
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

for f in $data/feats.scp $srcdir/final.ie $srcdir/final.dubm $srcdir/global_cmvn.stats $srcdir/splice_opts \
  $lang/phones.txt $srcdir/online_cmvn.conf $srcdir/final.mat; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $dir/log
silphonelist=$(cat $lang/phones/silence.csl) || exit 1;


sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

echo $ivector_period > $dir/ivector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)

gmm_feats="ark,s,cs:apply-cmvn-online --spk2utt=ark:$sdata/JOB/spk2utt --config=$srcdir/online_cmvn.conf $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
feats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

# This adds online-cmvn in $feats, upon request (configuration taken from UBM),
[ -f $srcdir/online_cmvn_iextractor ] && feats="$gmm_feats"

if [ $stage -le 1 ]; then
  if [ ! -z "$transform_mat" ]; then
    $cmd --num-threads $num_threads JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      ivector-extract --num-threads=$num_threads --acoustic-weight=$posterior_scale --compute-objf-change=true \
        --max-count=$max_count \
      $srcdir/final.ie "$feats" ark,s,cs:- ark:- \| \
      ivector-transform $transform_mat ark:- ark,t:$dir/ivectors_utt.JOB.ark || exit 1;
  else
    $cmd --num-threads $num_threads JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      ivector-extract --num-threads=$num_threads --acoustic-weight=$posterior_scale --compute-objf-change=true \
        --max-count=$max_count \
      $srcdir/final.ie "$feats" ark,s,cs:- ark,t,scp:$dir/ivectors_utt.JOB.ark,$dir/ivectors_utt.JOB.scp || exit 1;

    cat $dir/ivectors_utt.*.scp > $dir/ivectors_utt.scp
  fi
fi

ivector_dim=$[$(head -n 1 $dir/ivectors_utt.1.ark | wc -w) - 3] || exit 1;
echo  "$0: iVector dim is $ivector_dim"

base_feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

start_dim=$base_feat_dim
end_dim=$[$base_feat_dim+$ivector_dim-1]
absdir=$(utils/make_absolute.sh $dir)

if [ $stage -le 2 ]; then
  # here, we are just using the original features in $sdata/JOB/feats.scp for
  # their number of rows; we use the select-feats command to remove those
  # features and retain only the iVector features.
  $cmd JOB=1:$nj $dir/log/duplicate_feats.JOB.log \
    append-vector-to-feats scp:$sdata/JOB/feats.scp ark:$dir/ivectors_utt.JOB.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=$ivector_period ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$absdir/ivector_online.JOB.ark,$absdir/ivector_online.JOB.scp || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi

steps/nnet2/get_ivector_id.sh $srcdir > $dir/final.ie.id || exit 1

echo "$0: done extracting (pseudo-online) iVectors to $dir using the extractor in $srcdir."

