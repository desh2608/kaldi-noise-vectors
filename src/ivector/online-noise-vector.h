// ivector/online-noise-vector.h

// Copyright 2020   Johns Hopkins University (author: Desh Raj)

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


#ifndef KALDI_IVECTOR_ONLINE_NOISE_VECTOR_H_
#define KALDI_IVECTOR_ONLINE_NOISE_VECTOR_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {

// forward declaration
class OnlineNoiseVector;

class OnlineNoisePrior {
 friend class OnlineNoiseVector;

 public:
  OnlineNoisePrior() { }

  explicit OnlineNoisePrior(const OnlineNoisePrior &other):
    mu_n_(other.mu_n_),
    a_(other.a_),
    B_(other.B_),
    Lambda_n_(other.Lambda_n_),
    Lambda_s_(other.Lambda_s_),
    r_s_(other.r_s_),
    r_n_(other.r_n_) {
  };

  OnlineNoisePrior &operator = (const OnlineNoisePrior &other) {
    return *this;
  }

  int32 Dim() const;

  /// Takes the mean and covariance matrix computed from the
  /// training data and estimates the prior parameters.
  void EstimatePriorParameters(const VectorBase<BaseFloat> &mean,
                               const SpMatrix<BaseFloat> &covariance,
                               int32 dim, float scale);

  void EstimatePriorParameters(const VectorBase<BaseFloat> &mean,
                               const SpMatrix<BaseFloat> &covariance,
                               int32 dim,
                               Matrix<BaseFloat> &speech_var_sum,
                               Matrix<BaseFloat> &noise_var_sum,
                               int32 num_speech, int32 num_noise);
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

 protected:
  Vector<BaseFloat> mu_n_;  // mean of noise vectors.
  Vector<BaseFloat> a_;  // shift factor for mean of speech vectors.
  Matrix<BaseFloat> B_;  // scale factor for mean of speech vectors.
  Matrix<BaseFloat> Lambda_n_; // precision matrix for noise.
  Matrix<BaseFloat> Lambda_s_; // precision matrix for speech.
  double r_s_; // scaling factor for speech.
  double r_n_; // scaling factor for noise.

};

/// This class is used to extract online noise vectors. It is
/// initialized with an OnlineNoisePrior object and subsequently
/// generates online noise vectors by taking feats for an input
/// utterance and speech/silence targets.

class OnlineNoiseVector {
 public:
  /// Constructor. It is initialized with an OnlineNoisePrior object.
  /// Ideally you would want to initalize this once for each speaker,
  /// so that the updated scaling parameters can be reused in
  /// all utterances of the speaker.
  explicit OnlineNoiseVector(const OnlineNoisePrior &noise_prior, 
                             const int32 period);

  /// This function performs the actual noise vector computation, and
  /// can be called from a binary.
  void ExtractVectors(const Matrix<BaseFloat> &feats,
                      const std::vector<bool> &silence_decisions,
                      Matrix<BaseFloat> *noise_vectors);

  /// This function just computes the noise vectors from the
  /// prior parameters since no silence decisions are provided.
  void ExtractVectors(const Matrix<BaseFloat> &feats,
                      Matrix<BaseFloat> *noise_vectors);

  virtual ~OnlineNoiseVector();

 private:

  // This function updates current_nvector_  (which is our present estimate)
  // of the  current value for the n-vector, after a new chunk of 
  // data is seen. It takes as argument the silence decisions made by the 
  // GmmDecoder.
  void UpdateVector(
      SubMatrix<BaseFloat> &feats,
      std::vector<bool> &silence_decisions);

  // This function updates the scaling parameters r_s and r_n of the 
  // noise estimation model. This is done by maximizing the EM
  // objective. The derivation is not shown here.
  void UpdateScalingParams(
      SubMatrix<BaseFloat> &feats,
      std::vector<bool> &silence_decisions);

  // This stores the prior parameters that were used to initialize
  // the noise vectors.
  OnlineNoisePrior prior_;

  // This is similar to the ivector_period option used in online
  // ivectors, i.e., it determines the chunk size for which
  // noise vectors are computed.
  int32 period_;

  int32 dim_;

  // This is the current estimate of the noise vector
  Vector<BaseFloat> current_vector_;

  // Online statistic estimate
  int32 num_speech_;
  int32 num_noise_;
  Vector<BaseFloat> speech_sum_;
  Vector<BaseFloat> noise_sum_;
  Matrix<BaseFloat> speech_var_;
  Matrix<BaseFloat> noise_var_;
};


}  // namespace kaldi

#endif  // KALDI_IVECTOR_ONLINE_NOISE_VECTOR_H_
