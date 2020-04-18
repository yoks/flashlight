/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>

#include <arrayfire.h>
#include <dnnl.hpp>

#include <flashlight/common/Defines.h>

namespace fl {
namespace detail {

/**
 * A singleton class that contains a static instance of a mkldnn::engine.
 */
class MkldnnEngine {
 public:
  MkldnnEngine() : engine_(dnnl::engine::kind::cpu, 0) {}
  ~MkldnnEngine() = default;

  /// Prohibit assignment
  MkldnnEngine& operator=(MkldnnEngine const& e) = delete;

  dnnl::engine& getEngine();

  static MkldnnEngine& getInstance();

 private:
  dnnl::engine engine_;
};

/**
 * A singleton class that contains a static instance of a mkldnn::stream.
 */
class MkldnnStream {
 public:
  MkldnnStream() : stream_(fl::detail::MkldnnEngine::getInstance().getEngine()) {}
  ~MkldnnStream() = default;

  /// Prohibit assignment
  MkldnnStream& operator=(MkldnnStream const& s) = delete;

  dnnl::stream& getStream();

  static MkldnnStream& getInstance();

 private:
  dnnl::stream stream_;
};

/**
 * Helper for converting an ArrayFire af::dim4 into an MKL-DNN-compatible input
 * for mkldnn::memory::dims.
 */
dnnl::memory::dims convertAfToMklDnnDims(const std::vector<dim_t>& dims);

/**
 * Given some an mkldnn network (a ``std::vector<mkldnn::primitive>``), a
 * ``mkldnn::memory`` with some ordering, and a
 * ``mkldnn::memory::primitive_desc``, determines whether or not the memory
 * needs to be ordered based on the primitive descriptor's required ordering.
 *
 * If so, adds a ``mkldnn::reorder`` layer to the network, and returns a new
 * memory descriptor that will be properly reordered.
 */
dnnl::memory mkldnnAlignOrdering(
    std::vector<dnnl::primitive>& net,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc);

/**
 * Given a flashlight pooling mode, returns the corresponding mkldnn pooling
 * mode.
 */
dnnl::algorithm mkldnnMapToPoolingMode(const PoolingMode mode);

/**
 * Maps an ArrayFire array datatype into the corresponding MKL-DNN datatype.
 *
 * Needs to be explicitly inlined due to a bug with MKL-DNN.
 */
inline dnnl::memory::data_type mkldnnMapToType(const af::dtype t) {
  if (t == af::dtype::f32) {
    return dnnl::memory::data_type::f32;
  } else if (t == af::dtype::f64) {
    throw std::invalid_argument("float64 is not supported by MKL-DNN");
  } else {
    throw std::invalid_argument("data type not supported with MKL-DNN");
  }
}

} // namespace detail
} // namespace fl
