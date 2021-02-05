// ivector/online-noise-vector.cc

// Copyright 2020  Johns Hopkins University (author: Desh Raj)

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

#include "ivector/online-noise-vector.h"

namespace kaldi {

void OnlineNoisePrior::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineNoisePrior>");
  mu_n_.Write(os, binary);
  a_.Write(os, binary);
  B_.Write(os, binary);
  Lambda_n_.Write(os, binary);
  Lambda_s_.Write(os, binary);
  WriteBasicType(os, binary, r_s_);
  WriteBasicType(os, binary, r_n_);
  WriteToken(os, binary, "</OnlineNoisePrior>");
}

void OnlineNoisePrior::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineNoisePrior>");
  mu_n_.Read(is, binary);
  a_.Read(is, binary);
  B_.Read(is, binary);
  Lambda_n_.Read(is, binary);
  Lambda_s_.Read(is, binary);
  ReadBasicType(is, binary, &r_s_);
  ReadBasicType(is, binary, &r_n_);
  ExpectToken(is, binary, "</OnlineNoisePrior>");
}

int32 OnlineNoisePrior::Dim() const {
  return 2*a_.Dim();
}

void OnlineNoisePrior::EstimatePriorParameters(
    const VectorBase<BaseFloat> &mean,
    const SpMatrix<BaseFloat> &covariance,
    int32 dim, float scale) {
  SubVector<BaseFloat> mu_s(mean, 0, dim/2);
  SubVector<BaseFloat> mu_n(mean, dim/2, dim/2);
  Matrix<BaseFloat> Lambda(covariance), Cov(covariance);
  Lambda.Invert();
  SubMatrix<BaseFloat> Lambda_ss(Lambda, 0, dim/2, 0, dim/2); 
  SubMatrix<BaseFloat> Lambda_sn(Lambda, 0, dim/2, dim/2, dim/2); 
  SubMatrix<BaseFloat> Lambda_ns(Lambda, dim/2, dim/2, 0, dim/2); 
  SubMatrix<BaseFloat> Lambda_nn(Lambda, dim/2, dim/2, dim/2, dim/2);
  mu_n_ = mu_n;
  SubMatrix<BaseFloat> Cov_nn_inv(Cov, dim/2, dim/2, dim/2, dim/2);
  Cov_nn_inv.Invert();
  Lambda_n_ = Cov_nn_inv;
  Lambda_s_ = Lambda_ss;
  Matrix<BaseFloat> Lambda_sn_(Lambda_sn), Lambda_ss_inv(Lambda_ss);
  Lambda_ss_inv.Invert();
  // B = - (Lambda_ss_inv)^-1 Lambda_sn
  Matrix<BaseFloat> temp(dim/2, dim/2);
  temp.AddMatMat(-1.0, Lambda_ss_inv, kNoTrans, Lambda_sn_, kNoTrans, 0);
  B_ = temp;
  // a = mu_s - B mu_n
  a_ = mu_s;
  a_.AddMatVec(-1.0, temp, kNoTrans, mu_n_, 1);
  r_s_ = scale;
  r_n_ = scale;
}

void OnlineNoisePrior::EstimatePriorParameters(
    const VectorBase<BaseFloat> &mean,
    const SpMatrix<BaseFloat> &covariance,
    int32 dim,
    Matrix<BaseFloat> &speech_var_sum,
    Matrix<BaseFloat> &noise_var_sum,
    int32 num_speech, int32 num_noise) {
  EstimatePriorParameters(mean, covariance, dim, 1.0);
  if (num_speech > 0) { 
    r_s_ = (dim * num_speech) / TraceMatMat(Lambda_s_, speech_var_sum);
  }
  if (num_noise > 0) {
    r_n_ = (dim * num_noise) / TraceMatMat(Lambda_n_, noise_var_sum);
  }

}

OnlineNoiseVector::OnlineNoiseVector(
    const OnlineNoisePrior &noise_prior,
    const int32 period):
    period_(period), num_speech_(0), num_noise_(0) {
  dim_ = noise_prior.Dim();
  current_vector_ = Vector<BaseFloat>(dim_);
  prior_.mu_n_ = noise_prior.mu_n_;
  prior_.a_ = noise_prior.a_;
  prior_.B_ = noise_prior.B_;
  prior_.Lambda_n_ = noise_prior.Lambda_n_;
  prior_.Lambda_s_ = noise_prior.Lambda_s_;
  prior_.r_s_ = noise_prior.r_s_;
  prior_.r_n_ = noise_prior.r_n_;
  // initialize statistic variables
  speech_sum_.Resize(dim_/2);
  noise_sum_.Resize(dim_/2);
  speech_var_.Resize(dim_/2, dim_/2);
  noise_var_.Resize(dim_/2, dim_/2);
}

void OnlineNoiseVector::ExtractVectors(
    const Matrix<BaseFloat> &feats,
    const std::vector<bool> &silence_decisions,
    Matrix<BaseFloat> *noise_vectors) {
  int32 num_vectors = (feats.NumRows() + period_ - 1)/period_, num_done = 0;
  noise_vectors->Resize(num_vectors, dim_);
  for (int32 i = 0; i < num_vectors; ++i) {
    int32 num_rows = std::min(period_, feats.NumRows() - num_done);
    SubMatrix<BaseFloat> cur_feats(feats, i*period_, num_rows, 0, dim_/2);
    std::vector<bool>::const_iterator first = silence_decisions.begin() + i*period_;
    std::vector<bool>::const_iterator last = silence_decisions.begin() + i*period_ + num_rows;
    std::vector<bool> cur_decisions(first, last);
    UpdateVector(cur_feats, cur_decisions);
    UpdateScalingParams(cur_feats, cur_decisions);
    noise_vectors->CopyRowFromVec(current_vector_, i);
    num_done += num_rows;
  }
}

void OnlineNoiseVector::ExtractVectors(
    const Matrix<BaseFloat> &feats,
    Matrix<BaseFloat> *noise_vectors) {
  int32 num_rows = (feats.NumRows() + period_ - 1)/period_;
  noise_vectors->Resize(num_rows, dim_);
  Vector<BaseFloat> noise_vec(dim_);
  SubVector<BaseFloat> speech_mean(noise_vec, 0, dim_/2);
  SubVector<BaseFloat> sil_mean(noise_vec, dim_/2, dim_/2);
  sil_mean.AddVec(1.0, prior_.mu_n_);
  speech_mean.AddVec(1.0, prior_.a_);
  speech_mean.AddMatVec(1.0, prior_.B_, kNoTrans, prior_.mu_n_, 1.0);
  for (int32 i = 0; i < num_rows; ++i) {
    noise_vectors->CopyRowFromVec(noise_vec, i);
  }
}

void OnlineNoiseVector::UpdateVector(
    SubMatrix<BaseFloat> &feats,
    std::vector<bool> &silence_decisions) {
  // We first compute the sufficient statistics for the new
  // chunk of data (i.e., for which we have silence decisions
  // in silence_frames. We need, for both speech and noise
  // frames, the number of frames, sum of all frames, and
  // the variance of all frames.
  int32 dim = dim_/2;
  Vector<BaseFloat> cur_frame(dim);

  for (int32 i = 0; i < feats.NumRows(); ++i) {
    Vector<BaseFloat> cur_vec(dim);
    cur_vec.CopyFromVec(feats.Row(i));
    if (silence_decisions[i] == true) {
      // This is a silence frame
      num_noise_++;
      noise_sum_.AddVec(1.0, cur_vec);
      noise_var_.AddVecVec(1.0, cur_vec, cur_vec);  
    } else {
      // This is a speech frame
      num_speech_++;
      speech_sum_.AddVec(1.0, cur_vec);
      speech_var_.AddVecVec(1.0, cur_vec, cur_vec);
    }
  }

  // See paper for the math for this estimation method
  Matrix<BaseFloat> K(2*dim, 2*dim);
  Vector<BaseFloat> Q(2*dim);
  
  // Computing the matrix K
  SubMatrix<BaseFloat> K_11(K, 0, dim, 0, dim), K_12(K, 0, dim, dim, dim),
    K_21(K, dim, dim, 0, dim), K_22(K, dim, dim, dim, dim);
  K_11.AddMat(1.0 + prior_.r_s_*num_speech_, prior_.Lambda_s_);
  K_12.AddMatMat(-1.0, prior_.Lambda_s_, kNoTrans, prior_.B_, kNoTrans, 0);
  K_21.AddMatMat(-1.0, prior_.B_, kTrans, prior_.Lambda_s_, kNoTrans, 0);
  K_22.AddMat(1.0 + prior_.r_n_*num_noise_, prior_.Lambda_n_);
  {
    Matrix<BaseFloat> temp(dim, dim);
    temp.AddMatMat(1.0, prior_.B_, kTrans, prior_.Lambda_s_, kNoTrans, 0);
    K_22.AddMatMat(1.0, temp, kNoTrans, prior_.B_, kNoTrans, 1);
  }

  // Computing the vector Q
  SubVector<BaseFloat> Q_1(Q, 0, dim), Q_2(Q, dim, dim);
  {
    Vector<BaseFloat> temp = prior_.a_;
    temp.AddVec(prior_.r_s_, speech_sum_);
    Q_1.AddMatVec(1.0, prior_.Lambda_s_, kNoTrans, temp, 0.0);
  }
  {
    Vector<BaseFloat> temp = prior_.mu_n_;
    temp.AddVec(prior_.r_n_, noise_sum_);
    Q_2.AddMatVec(1.0, prior_.Lambda_n_, kNoTrans, temp, 0.0);
    temp.AddMatVec(1.0, prior_.Lambda_s_, kNoTrans, prior_.a_, 0.0);
    Q_2.AddMatVec(1.0, prior_.B_, kTrans, temp, 1.0);
  }

  // Compute the nvector from K and Q
  K.Invert();
  current_vector_.AddMatVec(1.0, K, kNoTrans, Q, 0.0);
}

void OnlineNoiseVector::UpdateScalingParams(
    SubMatrix<BaseFloat> &feats,
    std::vector<bool> &silence_decisions) {
  int32 dim = dim_/2;
  
  Vector<BaseFloat> cur_frame(dim);
  
  if (num_speech_ > 0) { 
    prior_.r_s_ = (dim * num_speech_) / 
      TraceMatMat(prior_.Lambda_s_, speech_var_);
  }
  if (num_noise_ > 0) {
  prior_.r_n_ = (dim * num_noise_) / 
    TraceMatMat(prior_.Lambda_n_, noise_var_);
  }
}

OnlineNoiseVector::~OnlineNoiseVector() {
  // Delete objects owned here.
}

}  // namespace kaldi
