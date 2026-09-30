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

// micro_test.h counterpart of near_nan_test.cc: its float checks must fail
// when exactly one side is NaN. see AmbiqAI/helia-rt#335

#include <limits>

#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Clears the failure the probed check recorded and returns whether it did.
bool TakeFailure() {
  const bool failed = micro_test::did_test_fail;
  micro_test::did_test_fail = false;
  return failed;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LegacyNearFailsWhenOneSideIsNaN) {
  TF_LITE_MICRO_EXPECT_NEAR(kNaN, 1.0f, 0.5f);
  const bool near_first = TakeFailure();
  TF_LITE_MICRO_EXPECT_NEAR(1.0f, kNaN, 0.5f);
  const bool near_second = TakeFailure();

  const float values[] = {kNaN, 1.0f};
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 0, values, 1, 0.5f);
  const bool array_first = TakeFailure();
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 1, values, 0, 0.5f);
  const bool array_second = TakeFailure();

  TF_LITE_MICRO_EXPECT(near_first);
  TF_LITE_MICRO_EXPECT(near_second);
  TF_LITE_MICRO_EXPECT(array_first);
  TF_LITE_MICRO_EXPECT(array_second);
}

TF_LITE_MICRO_TEST(LegacyFloatEqFailsWhenOneSideIsNaN) {
  TF_LITE_MICRO_EXPECT_EQ(kNaN, 0.0f);
  const bool eq_first = TakeFailure();
  TF_LITE_MICRO_EXPECT_EQ(0.0f, kNaN);
  const bool eq_second = TakeFailure();

  // TF_LITE_MICRO_ASSERT_EQ leaves the enclosing loop on failure.
  for (int i = 0; i < 1; ++i) {
    TF_LITE_MICRO_ASSERT_EQ(kNaN, 0.0f);
  }
  const bool assert_eq = TakeFailure();

  TF_LITE_MICRO_EXPECT(eq_first);
  TF_LITE_MICRO_EXPECT(eq_second);
  TF_LITE_MICRO_EXPECT(assert_eq);
}

TF_LITE_MICRO_TEST(LegacyNearFailsOutsideEpsilon) {
  TF_LITE_MICRO_EXPECT_NEAR(1.0f, 2.0f, 0.5f);
  const bool near_finite = TakeFailure();
  TF_LITE_MICRO_EXPECT_NEAR(kInf, 1.0f, 0.5f);
  const bool near_inf = TakeFailure();

  const float values[] = {1.0f, 2.0f, kInf};
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 0, values, 1, 0.5f);
  const bool array_finite = TakeFailure();
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 2, values, 0, 0.5f);
  const bool array_inf = TakeFailure();

  TF_LITE_MICRO_EXPECT(near_finite);
  TF_LITE_MICRO_EXPECT(near_inf);
  TF_LITE_MICRO_EXPECT(array_finite);
  TF_LITE_MICRO_EXPECT(array_inf);
}

TF_LITE_MICRO_TEST(LegacyFloatChecksPass) {
  const float values[] = {kNaN, kNaN, kInf, kInf};
  TF_LITE_MICRO_EXPECT_NEAR(kNaN, kNaN, 0.5f);
  TF_LITE_MICRO_EXPECT_NEAR(kInf, kInf, 0.5f);
  TF_LITE_MICRO_EXPECT_NEAR(1.0f, 1.25f, 0.5f);
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 0, values, 1, 0.5f);
  TF_LITE_MICRO_ARRAY_ELEMENT_EXPECT_NEAR(values, 2, values, 3, 0.5f);
  TF_LITE_MICRO_EXPECT_EQ(kNaN, kNaN);
  TF_LITE_MICRO_EXPECT_EQ(kInf, kInf);
  TF_LITE_MICRO_EXPECT_EQ(1.0f, 1.0f);
  TF_LITE_MICRO_EXPECT_NE(kNaN, 0.0f);
  TF_LITE_MICRO_ASSERT_EQ(kNaN, kNaN);
  TF_LITE_MICRO_ASSERT_EQ(kInf, kInf);
}

TF_LITE_MICRO_TESTS_END
