/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Installs the arena-exhaustion sweep as KernelRunner's InitAndPrepare hook.
// helia.inc links this into test and example binaries only, through
// MICROLITE_TEST_RUNTIME_SRCS, so no library archive carries it.
// see AmbiqAI/helia-rt#449

#include "tensorflow/lite/micro/kernels/helia/tests/allocation_sweep.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"

namespace {

[[maybe_unused]] const bool kAllocationSweepInstalled =
    (tflite::micro::KernelRunner::init_and_prepare_hook =
         tflite::micro::helia::SweepAllocationFailures,
     true);

}  // namespace
