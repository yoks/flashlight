/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/autograd/backend/cpu/MkldnnUtils.h"

#include <flashlight/common/Defines.h>

namespace fl {
namespace detail {

dnnl::stream& MkldnnStream::getStream() {
  return stream_;
}

MkldnnStream& MkldnnStream::getInstance() {
  static MkldnnStream instance;
  return instance;
}

dnnl::engine& MkldnnEngine::getEngine() {
  return engine_;
}

MkldnnEngine& MkldnnEngine::getInstance() {
  static MkldnnEngine instance;
  return instance;
}

dnnl::memory::dims convertAfToMklDnnDims(const std::vector<dim_t>& afDims) {
  // MKL-DNN uses ints in dims
  std::vector<dnnl::memory::dim> intVec(afDims.begin(), afDims.end());
  return dnnl::memory::dims(intVec);
}

dnnl::memory mkldnnAlignOrdering(
    std::vector<dnnl::primitive>& net,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc) {
  auto memoryOut = memory;
  if (memory.get_desc() != desc) {
    memoryOut =
        dnnl::memory(desc, fl::detail::MkldnnEngine::getInstance().getEngine()); // use the ordering requested by the descriptor
    net.push_back(dnnl::reorder(memory, memoryOut));
  }
  return memoryOut;
}

dnnl::algorithm mkldnnMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return dnnl::algorithm::pooling_max;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_include_padding;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_exclude_padding;
    default:
      throw std::invalid_argument("unsupported pooling mode for cuDNN");
  }
}

} // namespace detail
} // namespace fl
