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
        "Compute a noise vector for each utterance, by\n"
        "taking average of frames segmented into silence and garbage,\n"
        "and mean of speech frames, and concatenating. If no target specifier\n"
        "is provided, then it returns average over all frames in\n"
        "the utterance."
        "Usage: compute-noise-vector [options] <feats-rspecifier> "
        " <targets-rspecifier> <vector-wspecifier>\n"
        "E.g.: compute-noise-vector [options] scp:feats.scp scp:targets.scp ark:-\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier, target_rspecifier, vector_wspecifier;
    feat_rspecifier = po.GetArg(1),
      target_rspecifier = po.GetArg(2),
      vector_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);
    RandomAccessBaseFloatMatrixReader target_reader(target_rspecifier);

    int32 num_done = 0, num_err = 0;

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      Vector<BaseFloat> speech_feat(feat.NumCols());
      Vector<BaseFloat> noise_feat(feat.NumCols());
      int32 num_speech = 0, num_noise = 0;
      
      if (!target_reader.HasKey(utt)) {
        KALDI_WARN << "No target found for utterance. Creating vector of 0s. " << utt;
        num_err++;
      } else {
        const Matrix<BaseFloat> &target = target_reader.Value(utt);

        if (feat.NumRows() != target.NumRows()) {
          KALDI_WARN << "Mismatch in number for frames " << feat.NumRows()
                     << " for features and targets " << target.NumRows()
                     << ", for utterance " << utt
                     << ". Creating vector of 0s.";
          num_err++;
        } else {
          for (int32 i = 0; i < feat.NumRows(); i++) {
            if (target(i,1) > target(i,0) && target(i,1) > target(i,2)) {
              speech_feat.AddVec(1.0, feat.Row(i));
              num_speech += 1;
            } else {
              noise_feat.AddVec(1.0, feat.Row(i));
              num_noise += 1;
            }
          }
        }
      }
      if (num_speech > 0) { speech_feat.Scale(1.0/num_speech); }
      if (num_noise > 0) { noise_feat.Scale(1.0/num_noise); }

      speech_feat.Resize(2*feat.NumCols(), kCopyData);
      SubVector<BaseFloat> noise_subfeat(speech_feat, feat.NumCols(), feat.NumCols());
      noise_subfeat.CopyFromVec(noise_feat);
      vector_writer.Write(utt, speech_feat);
      num_done++;
    }

    KALDI_LOG << "Done computing noise vectors; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
