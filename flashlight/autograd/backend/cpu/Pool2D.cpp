/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <arrayfire.h>
#include <dnnl.h>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Utils.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cpu/MkldnnUtils.h"
#include "flashlight/common/DevicePtr.h"

using namespace dnnl;

namespace {

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

namespace fl {

Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode) {
  auto inputDimsRaw = input.dims();
  auto output = af::array(
      1 + (input.dims(kWIdx) + 2 * px - wx) / sx,
      1 + (input.dims(kHIdx) + 2 * py - wy) / sy,
      input.dims(kChannelSizeIdx),
      input.dims(kBatchSizeIdx));

  // Dims
  memory::dims inputDims =
      detail::convertAfToMklDnnDims({input.dims(kBatchSizeIdx),
                                     input.dims(kChannelSizeIdx),
                                     input.dims(kHIdx),
                                     input.dims(kWIdx)});
  memory::dims outputDims =
      detail::convertAfToMklDnnDims({input.dims(kBatchSizeIdx),
                                     input.dims(kChannelSizeIdx),
                                     output.dims(kHIdx),
                                     output.dims(kWIdx)});
  memory::dims windowDims = {wy, wx};
  memory::dims strideDims = {sy, sx};
  memory::dims paddingDims = {py, px};

  auto dataType = detail::mkldnnMapToType(input.type());
  auto formatNCHW = memory::format_tag::nchw;
  auto formatAny = memory::format_tag::any;

  // Memory desc
  auto inputMD = memory::desc({inputDims}, dataType, formatNCHW);
  auto outputMD = memory::desc({outputDims}, dataType, formatAny);

  // Memory
  auto mkldnnEngine = detail::MkldnnEngine::getInstance().getEngine();
  DevicePtr inputRaw(input.array());
  auto inputMemoryInit =
      memory({{inputDims}, dataType, formatNCHW}, mkldnnEngine, inputRaw.get());
  DevicePtr outputRaw(output);
  auto outputMemoryInit = memory(
      {{outputDims}, dataType, formatNCHW}, mkldnnEngine, outputRaw.get());

  // Choose a mode based on whether gradients are needed
  auto forwardMode = input.isCalcGrad() ? prop_kind::forward_training
                                        : prop_kind::forward_inference;

  // Descriptors
  auto poolingMode = detail::mkldnnMapToPoolingMode(mode);
  auto desc = pooling_forward::desc(
      forwardMode,
      poolingMode,
      inputMD,
      outputMD,
      strideDims,
      windowDims,
      paddingDims,
      paddingDims);
  auto primDesc = pooling_forward::primitive_desc(desc, mkldnnEngine);

  // Network
  std::vector<primitive> network;
  // Reorder if needed
  auto inputPrimDesc = primDesc.src_desc();
  auto outputPrimDesc = primDesc.dst_desc();
  auto inputMemory =
      detail::mkldnnAlignOrdering(network, inputMemoryInit, inputPrimDesc);
  auto outputMemory = outputMemoryInit;
  if (outputMemoryInit.get_desc() != outputPrimDesc) {
    outputMemory = memory(outputPrimDesc, mkldnnEngine);
  }
  // Workspace and layer (only training mode requires a workspace)
  std::shared_ptr<memory> workspaceMemory; // no default ctors
  std::shared_ptr<pooling_forward> pooling;
  pooling = std::make_shared<pooling_forward>(primDesc);

  std::unordered_map<int, dnnl::memory> input_args;
  input_args.insert({DNNL_ARG_SRC, inputMemory});
  input_args.insert({DNNL_ARG_DST, outputMemory});

  if (input.isCalcGrad()) {
    workspaceMemory = std::make_shared<memory>(primDesc.workspace_desc(), mkldnnEngine);
    input_args.insert({DNNL_ARG_WORKSPACE, *workspaceMemory});
  }

  network.push_back(*pooling);

  // Add output reordering if needed
  if (outputMemory.get_desc() != outputMemoryInit.get_desc()) {
    network.push_back(dnnl::reorder(outputMemory, outputMemoryInit));
  }

  for (size_t i = 0; i < network.size(); ++i) {
    network.at(i).execute(
        detail::MkldnnStream::getInstance().getStream(), input_args);
  }

  pooling->execute(detail::MkldnnStream::getInstance().getStream(), input_args);

  auto gradFunc =
      [dataType,
       formatNCHW,
       inputDimsRaw, // need to pass if inputs are empty
       primDesc, // forward desc
       poolingMode,
       // needed for backwards pass. null in inference mode
       workspaceMemory,
       // dims
       inputDims,
       outputDims,
       windowDims,
       strideDims,
       paddingDims,
       // capture the output memory primitive desc for reordering, since it
       // can't be retrieved from the pooling primitive descriptor
       outputMemory](
          std::vector<Variable>& inputs, const Variable& grad_output) {
        auto& in = inputs[0];
        if (!in.isCalcGrad()) {
          return;
        }

        auto gradInput =
            Variable(af::array(inputDimsRaw, af::dtype::f32), false);
        auto mkldnnEngineBwd = detail::MkldnnEngine::getInstance().getEngine();

        // Memory
        DevicePtr gradInputRaw(gradInput.array());
        auto gradInputMemoryInit = memory(
            {{inputDims}, dataType, formatNCHW},
            mkldnnEngineBwd,
            gradInputRaw.get());
        DevicePtr gradOutputRaw(grad_output.array());
        auto gradOutputMemoryInit = memory(
            {{outputDims}, dataType, formatNCHW},
            mkldnnEngineBwd,
            gradOutputRaw.get());

        // Descriptors
        // Memory descriptors from initialized memory must be used since
        // pooling_backward descriptors require an ordering
        auto gradInputMD = gradInputMemoryInit.get_desc();
        auto gradOutputMD = gradOutputMemoryInit.get_desc();
        auto bwdDesc = pooling_backward::desc(
            poolingMode,
            gradInputMD,
            gradOutputMD,
            strideDims,
            windowDims,
            paddingDims,
            paddingDims);
        // Pass forward descriptor as a hint
        auto bwdPrimDesc = pooling_backward::primitive_desc(
            bwdDesc, mkldnnEngineBwd, primDesc);

        std::vector<primitive> networkBackward;
        // Reorder output memory if required
        auto gradOutputMemory = detail::mkldnnAlignOrdering(
            networkBackward, gradOutputMemoryInit, outputMemory.get_desc());

        std::unordered_map<int, dnnl::memory> input_args;
        input_args.insert({DNNL_ARG_DIFF_SRC, gradInputMemoryInit});
        input_args.insert({DNNL_ARG_DIFF_DST, gradOutputMemory});
        input_args.insert({DNNL_ARG_WORKSPACE, *workspaceMemory});

        auto poolBwd = pooling_backward(bwdPrimDesc);
        networkBackward.push_back(poolBwd);

        for (size_t i = 0; i < networkBackward.size(); ++i) {
          networkBackward.at(i).execute(
              detail::MkldnnStream::getInstance().getStream(), input_args);
        }
        in.addGrad(gradInput);
      };

  return Variable(output, {input.withoutData()}, gradFunc);
}

} // namespace fl
