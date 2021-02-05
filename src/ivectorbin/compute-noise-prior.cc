// ivectorbin/compute-noise-prior.cc

// Copyright 2019  Johns Hopkins University (Author: Desh Raj)

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
#include "ivector/online-noise-vector.h"

namespace kaldi {

/* This code implements a Bayesian model for online estimation of speech
 * and noise vectors. First, we estimate the prior parameters from the 
 * training data:
 * pi = (mu_n, a, B, Lambda_n, Lambda_s),
 * where mu_n = mean of noise frames in training set,
 * a = mu_s + {Lambda_ss}^-1 Lambda_sn mu_n,
 * B = - {Lambda_ss}^-1 Lambda_sn,
 * Lambda's are the inverse of the covariance mattrices (i.e., they are the
 * precision matrices). 
 * For derivation, see related paper.
 *
 * After estimating the prior parameters, at inference time, the posteriors
 * are computed through an E-M like procedure. The related code can be found
 * in ivector/online-noise-vector.cc.
*/

void ComputeAndSubtractMean(
    std::map<std::string, Vector<BaseFloat> *> utt2vector,
    Vector<BaseFloat> *mean_out) {
  int32 dim = utt2vector.begin()->second->Dim();
  size_t num_vectors = utt2vector.size();
  Vector<BaseFloat> mean(dim);
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  for (iter = utt2vector.begin(); iter != utt2vector.end(); ++iter)
    mean.AddVec(1.0 / num_vectors, *(iter->second));
  mean_out->Resize(dim);
  mean_out->CopyFromVec(mean);
  for (iter = utt2vector.begin(); iter != utt2vector.end(); ++iter)
    iter->second->AddVec(-1.0, *mean_out);
}

void ComputeCovarianceMatrix(
    std::map<std::string, Vector<BaseFloat> *> &utt2vector,
    SpMatrix<BaseFloat> *covariance) {
  KALDI_ASSERT(!utt2vector.empty());
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  int32 N = utt2vector.size() - 1;
  for (iter = utt2vector.begin(); iter != utt2vector.end(); ++iter) {
    Vector<BaseFloat> noise_vec = *(iter->second);
    covariance->AddVec2(1.0/N, noise_vec);
  }
}
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes a Noise Prior object from a set of noise vectors computed from\n"
        "training data\n"
        "\n"
        "Usage:  compute-noise-prior [options] <noise-vector-rspecifier> "
        "<noise-prior-out>\n"
        "e.g.: \n"
        " compute-noise-prior ark:noise_vec.ark noise_prior\n";

    ParseOptions po(usage);

    bool binary = true;
    float scale = 1;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("scale", &scale, "Init value for r_s and r_n");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string noise_vec_rspecifier = po.GetArg(1),
        noise_prior_wxfilename = po.GetArg(2);

    int32 num_done = 0, num_err = 0, dim = 0;

    SequentialBaseFloatVectorReader noise_vec_reader(noise_vec_rspecifier);

    std::map<std::string, Vector<BaseFloat> *> utt2noise_vec;

    for (; !noise_vec_reader.Done(); noise_vec_reader.Next()) {
      std::string utt = noise_vec_reader.Key();
      const Vector<BaseFloat> noise_vector = noise_vec_reader.Value();
      if (utt2noise_vec.count(utt) != 0) {
        KALDI_WARN << "Duplicate noise vector found for utterance " << utt
                   << ", ignoring it.";
        num_err++;
        continue;
      }
      utt2noise_vec[utt] = new Vector<BaseFloat>(noise_vector);
      if (dim == 0) {
        dim = noise_vector.Dim();
      } else {
        KALDI_ASSERT(dim == noise_vector.Dim() && "Noise vector dimension mismatch");
      }      
      num_done++;
    }

    Vector<BaseFloat> mean;
    ComputeAndSubtractMean(utt2noise_vec, &mean);
    KALDI_LOG << "2-norm of noise vector mean is " << mean.Norm(2.0);

    SpMatrix<BaseFloat> covariance(dim);
    ComputeCovarianceMatrix(utt2noise_vec, &covariance);
    OnlineNoisePrior noise_prior;
    noise_prior.EstimatePriorParameters(mean, covariance, dim, scale);

    WriteKaldiObject(noise_prior, noise_prior_wxfilename, binary);
    KALDI_LOG << "Wrote OnlineNoisePrior parameters to "
              << PrintableWxfilename(noise_prior_wxfilename);
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
