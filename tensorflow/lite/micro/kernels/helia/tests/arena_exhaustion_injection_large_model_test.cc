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

// The allocation-failure injection of arena_exhaustion_injection_test.cc for
// person_detect and DTLN, in a binary of its own to keep each FVP run short.
// They fail each allocation only together with every later one.
// see AmbiqAI/helia-rt#449

#include "tensorflow/lite/micro/kernels/helia/tests/arena_sweep_models.h"

TEST(HeliaArenaExhaustionInjectionTest, PersonDetectModel) {
  InjectAllocationFailures("person_detect", g_person_detect_model_data,
                           /*resource_variables=*/false, /*alone_too=*/false);
}

TEST(HeliaArenaExhaustionInjectionTest, DtlnModel) {
  InjectAllocationFailures("dtln_noise_suppression",
                           g_dtln_noise_suppression_model_data,
                           /*resource_variables=*/false, /*alone_too=*/false);
}

TF_LITE_MICRO_TESTS_MAIN
