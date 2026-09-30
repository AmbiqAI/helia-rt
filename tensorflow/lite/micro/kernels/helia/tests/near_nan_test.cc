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

// The near and float-equal checks must fail when exactly one side is NaN;
// every float kernel golden depends on it. Two NaNs compare equal so tests
// that expect NaN propagation keep passing. see AmbiqAI/helia-rt#335

#include <limits>

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Clears the failure the probed check recorded and returns whether it did.
bool TakeFailure() {
  auto& runner = micro_test::internal::TestRunner::Get();
  const bool failed = runner.fail();
  runner.fail() = false;
  runner.fatal_fail() = false;
  runner.nonfatal_fail() = false;
  return failed;
}

void AssertNearNaN() { ASSERT_NEAR(kNaN, 1.0f, 0.5f); }
void AssertFloatEqNaN() { ASSERT_FLOAT_EQ(0.0f, kNaN); }

}  // namespace

TEST(NearNaN, NearFailsWhenOneSideIsNaN) {
  EXPECT_NEAR(kNaN, 1.0f, 0.5f);
  const bool nan_first = TakeFailure();
  EXPECT_NEAR(1.0f, kNaN, 0.5f);
  const bool nan_second = TakeFailure();
  EXPECT_NEAR(static_cast<double>(kNaN), 0.0, 1e-3);
  const bool nan_double = TakeFailure();

  EXPECT_TRUE(nan_first);
  EXPECT_TRUE(nan_second);
  EXPECT_TRUE(nan_double);
}

TEST(NearNaN, FloatEqFailsWhenOneSideIsNaN) {
  EXPECT_FLOAT_EQ(kNaN, 0.0f);
  const bool nan_first = TakeFailure();
  EXPECT_FLOAT_EQ(0.0f, kNaN);
  const bool nan_second = TakeFailure();

  EXPECT_TRUE(nan_first);
  EXPECT_TRUE(nan_second);
}

TEST(NearNaN, AssertFailsWhenOneSideIsNaN) {
  AssertNearNaN();
  const bool near = TakeFailure();
  AssertFloatEqNaN();
  const bool float_eq = TakeFailure();

  EXPECT_TRUE(near);
  EXPECT_TRUE(float_eq);
}

TEST(NearNaN, NearFailsOutsideEpsilon) {
  EXPECT_NEAR(1.0f, 2.0f, 0.5f);
  const bool finite = TakeFailure();
  EXPECT_NEAR(kInf, 1.0f, 0.5f);
  const bool inf = TakeFailure();

  EXPECT_TRUE(finite);
  EXPECT_TRUE(inf);
}

TEST(NearNaN, NearPasses) {
  EXPECT_NEAR(kNaN, kNaN, 0.5f);
  EXPECT_NEAR(kInf, kInf, 0.5f);
  EXPECT_NEAR(-kInf, -kInf, 0.5f);
  EXPECT_NEAR(1.0f, 1.25f, 0.5f);
  EXPECT_FLOAT_EQ(kNaN, kNaN);
  EXPECT_FLOAT_EQ(1.0f, 1.0f);
  ASSERT_NEAR(kNaN, kNaN, 0.5f);
}

TF_LITE_MICRO_TESTS_MAIN
