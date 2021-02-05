// ivectorbin/compute-noise-vector.cc

// Copyright   2019   Desh Raj

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute a NAT vector or each utterance, by\n"
        "taking average of first and last 10 frames (see Seltzer et al.)\n"
        "Usage: compute-noise-vector-seltzer [options] <feats-rspecifier> "
        " <vector-wspecifier>\n"
        "E.g.: compute-noise-vector-seltzer [options] scp:feats.scp ark:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier, vector_wspecifier;
    feat_rspecifier = po.GetArg(1),
      vector_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_done = 0, num_err = 0;

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      Vector<BaseFloat> noise_feat(feat.NumCols());
      int32 num_noise = 0;

      for (int32 i = 0; i < feat.NumRows(); i++) {
        if (i < 10 || i+10 >= feat.NumRows()) {
          noise_feat.AddVec(1.0, feat.Row(i));
          num_noise += 1;
        }
      }
      
      if (num_noise > 0) { noise_feat.Scale(1.0/num_noise); }

      vector_writer.Write(utt, noise_feat);
      num_done++;
    }

    KALDI_LOG << "Done computing NAT vectors; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
