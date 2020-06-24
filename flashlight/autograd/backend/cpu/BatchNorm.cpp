/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <arrayfire.h>
#include <dnnl.hpp>

#include <iostream>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cpu/MkldnnUtils.h"
#include "flashlight/common/DevicePtr.h"

namespace fl {

namespace {

// Flashlight accept HWCN order according to docs
constexpr size_t kHIdx = 0;
constexpr size_t kWIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double epsilon) {
  auto output = af::array(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  if (runningVar.isempty()) {
    runningVar = Variable(af::constant(1.0, nfeatures, input.type()), false);
  }

  if (runningMean.isempty()) {
    runningMean = Variable(af::constant(0.0, nfeatures, input.type()), false);
  }

  // Check if axes is valid
  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto dType = detail::mkldnnMapToType(input.type());
  auto mkldnnEngine = detail::MkldnnEngine::getInstance().getEngine();
  auto formatX = dnnl::memory::format_tag::x;
  auto format2d = dnnl::memory::format_tag::nc;
  // MKL-DNN requires NCHW order, and it thinks data are in ROW-MAJOR layout.
  // The input tesor is in WHCN order and layout in COLUMN-MAJOR (arrayfire).
  // Thus, MKL-DNN can access the required element correctly.
  auto formatNCHW = dnnl::memory::format_tag::nchw;

  /****************************************************************************/
  // Prepare combined weights

  // If empty, user specifies affine to false. Both not trainable.
  auto weightNonempty = weight.isempty()
      ? Variable(af::constant(1.0, nfeatures, input.type()), false)
      : weight;
  auto biasNonempty = bias.isempty()
      ? Variable(af::constant(0.0, nfeatures, input.type()), false)
      : bias;

  // MKLDNN only accept weight and bias as a combined input.
  // https://fburl.com/l0bctocp
  auto weightsMkldnn =
      af::join(0, weightNonempty.array(), biasNonempty.array());

  /****************************************************************************/
  // Prepare the fwd operator descriptor

  std::vector<int> rawDims;

  if (min_axis == 0) {
    rawDims = {1, 1, nfeatures, (int)input.elements() / nfeatures};
  } else {
    int batchsz = 1;
    for (int i = max_axis + 1; i < 4; ++i) {
      batchsz *= input.dims(i);
    }
    rawDims = {1, (int)input.elements() / (nfeatures * batchsz), nfeatures, batchsz};
  }

  auto inputOutputDims = detail::convertAfToMklDnnDims({
      rawDims[kBatchSizeIdx],
      rawDims[kChannelSizeIdx],
      rawDims[kHIdx],
      rawDims[kWIdx],
  });

  auto inputOutputMemDesc =
      dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);

  // https://fburl.com/6latj733
  auto flag = train ? dnnl::normalization_flags::none
                    : dnnl::normalization_flags::use_global_stats;
  flag = flag | dnnl::normalization_flags::use_scale_shift;

  // FWD primitive descriptor construction
  auto kind = train ? dnnl::prop_kind::forward_training
                    : dnnl::prop_kind::forward_inference;
  auto fwdDesc = dnnl::batch_normalization_forward::desc(
      kind, inputOutputMemDesc, epsilon, flag);
  auto fwdPrimDesc =
      std::make_shared<dnnl::batch_normalization_forward::primitive_desc>(
          fwdDesc, mkldnnEngine);

  /****************************************************************************/
  // Prepare memories

  // input
  DevicePtr inputRaw(input.array());
  auto inputMemDesc = dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto inputMemInit = dnnl::memory(inputMemDesc, mkldnnEngine, inputRaw.get());

  // out
  DevicePtr outputRaw(output);
  auto outputMemDesc = dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto outputMem = dnnl::memory(outputMemDesc, mkldnnEngine, outputRaw.get());

  // mean
  DevicePtr meanRaw(runningMean.array());
  auto meanDims = detail::convertAfToMklDnnDims({runningMean.dims(0)});
  auto meanMemDesc = dnnl::memory::desc({meanDims}, dType, formatX);
  auto meanMemInit = dnnl::memory(meanMemDesc, mkldnnEngine, meanRaw.get());

  // var
  DevicePtr varRaw(runningVar.array());
  auto varDims = detail::convertAfToMklDnnDims({runningVar.dims(0)});
  auto varMemDesc = dnnl::memory::desc({varDims}, dType, formatX);
  auto varMemInit = dnnl::memory(varMemDesc, mkldnnEngine, varRaw.get());

  // weightMKLDNN
  DevicePtr weightsMkldnnRaw(weightsMkldnn);
  auto weightsMkldnnDims = detail::convertAfToMklDnnDims({2, nfeatures});
  auto weightsMkldnnMemDesc =
      dnnl::memory::desc({weightsMkldnnDims}, dType, format2d);
  auto weightsMkldnnMemInit =
      dnnl::memory(weightsMkldnnMemDesc, mkldnnEngine, weightsMkldnnRaw.get());

  /****************************************************************************/
  // Setup primitive operator

  std::shared_ptr<dnnl::batch_normalization_forward> bn =
      std::make_shared<dnnl::batch_normalization_forward>(*fwdPrimDesc);

  /****************************************************************************/
  // Setup execution network

  std::unordered_map<int, dnnl::memory> bnorm_args;
  bnorm_args.insert({DNNL_ARG_SRC, inputMemInit});
  bnorm_args.insert({DNNL_ARG_MEAN, meanMemInit});
  bnorm_args.insert({DNNL_ARG_VARIANCE, varMemInit});
  bnorm_args.insert({DNNL_ARG_SCALE_SHIFT, weightsMkldnnMemInit});
  bnorm_args.insert({DNNL_ARG_DST, outputMem});

  bn->execute(detail::MkldnnStream::getInstance().getStream(), bnorm_args);

  /****************************************************************************/
  // Setup backward func

  auto gradFunc = [train,
                   epsilon,
                   nfeatures,
                   fwdPrimDesc,
                   outputMemDesc,
                   inputOutputDims,
                   formatNCHW,
                   format2d,
                   dType,
                   weightsMkldnn,
                   weightsMkldnnDims,
                   inputMemInit,
                   meanMemInit,
                   varMemInit,
                   weightsMkldnnMemInit](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    if (!train) {
      throw std::logic_error(
          "can't compute batchnorm grad when train was not specified");
    }

    auto mkldnnEngineBwd = detail::MkldnnEngine::getInstance().getEngine();

    auto& inputRef = inputs[0];
    auto weightRef = inputs[1].isempty()
        ? Variable(af::constant(1.0, nfeatures, inputRef.type()), false)
        : inputs[1];
    auto biasRef = inputs[2].isempty()
        ? Variable(af::constant(0.0, nfeatures, inputRef.type()), false)
        : inputs[2];
    ;

    auto grad_input =
        Variable(af::array(inputRef.dims(), inputRef.type()), false);

    auto grad_weightsMKLDNN =
        Variable(af::array(weightsMkldnn.dims(), weightsMkldnn.type()), false);

    /********************************************************************/
    // Prepare memories for grad_output
    DevicePtr gradOutputRaw(grad_output.array());
    auto gradOutputMemDesc =
        dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradOutputMemInit =
        dnnl::memory(gradOutputMemDesc, mkldnnEngineBwd, gradOutputRaw.get());

    DevicePtr gradInputRaw(grad_input.array());
    auto gradInputMemDesc =
        dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradInputMemInit =
        dnnl::memory(gradInputMemDesc, mkldnnEngineBwd, gradInputRaw.get());

    DevicePtr gradWeightsMkldnnRaw(grad_weightsMKLDNN.array());
    auto gradWeightsMkldnnMemDesc =
        dnnl::memory::desc({weightsMkldnnDims}, dType, format2d);
    auto gradWeightsMkldnnMemInit = dnnl::memory(
        gradWeightsMkldnnMemDesc, mkldnnEngineBwd, gradWeightsMkldnnRaw.get());

    /********************************************************************/
    // Setup backward descriptor:

    auto bwdDesc = dnnl::batch_normalization_backward::desc(
        dnnl::prop_kind::backward,
        gradOutputMemDesc,
        outputMemDesc,
        epsilon,
        dnnl::normalization_flags::use_scale_shift);

    /********************************************************************/
    // Setup backward prim descriptor:
    auto bwdPrimDesc = dnnl::batch_normalization_backward::primitive_desc(
        bwdDesc, mkldnnEngineBwd, *fwdPrimDesc);

    /********************************************************************/
    // Construct bwd op

    auto bwdPrim =
        std::make_shared<dnnl::batch_normalization_backward>(bwdPrimDesc);

    /********************************************************************/
    // Setup execution network
    std::unordered_map<int, dnnl::memory> bwd_args;
    bwd_args.insert({DNNL_ARG_SRC, inputMemInit});
    bwd_args.insert({DNNL_ARG_MEAN, meanMemInit});
    bwd_args.insert({DNNL_ARG_VARIANCE, varMemInit});
    bwd_args.insert({DNNL_ARG_SCALE_SHIFT, weightsMkldnnMemInit});
    bwd_args.insert({DNNL_ARG_DIFF_DST, gradOutputMemInit});
    bwd_args.insert({DNNL_ARG_DIFF_SRC, gradInputMemInit});
    bwd_args.insert({DNNL_ARG_DIFF_SCALE_SHIFT, gradWeightsMkldnnMemInit});

    bwdPrim->execute(detail::MkldnnStream::getInstance().getStream(), bwd_args);

    /********************************************************************/
    // Update grad

    inputRef.addGrad(grad_input);

    // extracting grads from grad_weightsMKLDNN for weight and bias
    if (weightRef.isCalcGrad()) {
      auto gradWeight = Variable(
          grad_weightsMKLDNN.array()(
              af::seq(0, nfeatures - 1), af::span, af::span, af::span),
          false);
      weightRef.addGrad(gradWeight);

      auto gradBias = Variable(
          grad_weightsMKLDNN.array()(
              af::seq(nfeatures, 2 * nfeatures - 1),
              af::span,
              af::span,
              af::span),
          false);
      if (!biasRef.isempty()) {
        biasRef.addGrad(gradBias);
      }
    }
  };

  /****************************************************************************/
  // return

  return Variable(output, {input, weight, bias}, gradFunc);
}

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  // CPU backend MKL-DNN does not support momentum factor.
  // If momentum enabled, throw error.
  if (momentum == 0.0) {
    return batchnorm(
        input, weight, bias, runningMean, runningVar, axes, train, epsilon);
  } else {
    throw std::runtime_error("BatchNorm CPU backend doesn't support momentum.");
  }
}

} // namespace fl
