#!/bin/bash

KALDI_DIR=$1
[ -z $KALDI_DIR ] && echo "Missing Kaldi path" && exit 1

KALDI_PATH=$(realpath ${KALDI_DIR})

# Link to steps and utils
ln -sfv ${KALDI_PATH}/egs/wsj/s5/steps .
ln -sfv ${KALDI_PATH}/egs/wsj/s5/utils .

echo "Creating path.sh with necessary paths"
cat <<EOF >path.sh
export KALDI_ROOT=${KALDI_PATH}
[ -f \$KALDI_ROOT/tools/env.sh ] && . \$KALDI_ROOT/tools/env.sh
export PATH=\$PWD/utils/:\$KALDI_ROOT/tools/openfst/bin:\$PWD:\$PATH
[ ! -f \$KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file \$KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. \$KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
EOF

# Link to local scripts (mainly for data preparation)
AURORA4_LOCAL_DIR=${KALDI_PATH}/egs/aurora4/s5/local

cd local
ln -sfv ${AURORA4_LOCAL_DIR}/aurora2flist.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/aurora4_data_prep.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/aurora4_format_data.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/cstr_ndx2flist.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/cstr_wsj_data_prep.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/cstr_wsj_extend_dict.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/dict .
ln -sfv ${AURORA4_LOCAL_DIR}/find_transcripts.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/flist2scp_12.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/flist2scp.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/ndx2flist.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/normalize_transcript.pl .
ln -sfv ${AURORA4_LOCAL_DIR}/cstr_wsj_extend_dict.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/wsj_format_local_lms.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/wsj_prepare_dict.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/wsj_train_lms.sh .
ln -sfv ${AURORA4_LOCAL_DIR}/score.sh .
cd ..

echo "Setup complete!"