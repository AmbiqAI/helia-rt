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

// For every model in the tree the helia build runs, each arena allocation
// AllocateTensors makes is failed in turn, together with every later one, as
// an exhausted arena does, and alone: AllocateTensors must return
// kTfLiteError each time without writing past the arena end. person_detect
// and DTLN are in arena_exhaustion_injection_large_model_test.cc.
// see AmbiqAI/helia-rt#449

#include "tensorflow/lite/micro/kernels/helia/tests/arena_sweep_models.h"

TEST(HeliaArenaExhaustionInjectionTest, KeywordModelPersistentDominated) {
  InjectAllocationFailures("keyword_scrambled", g_keyword_scrambled_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, Keyword8BitModel) {
  InjectAllocationFailures("keyword_scrambled_8bit",
                           g_keyword_scrambled_8bit_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, Int8ConvModelActivationDominated) {
  InjectAllocationFailures("test_conv_model", kTestConvModelData);
}

TEST(HeliaArenaExhaustionInjectionTest, HelloWorldModels) {
  InjectAllocationFailures("hello_world_float", g_hello_world_float_model_data);
  InjectAllocationFailures("hello_world_int8", g_hello_world_int8_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, MicroSpeechModels) {
  InjectAllocationFailures("audio_preprocessor_float",
                           g_audio_preprocessor_float_model_data);
  InjectAllocationFailures("audio_preprocessor_int8",
                           g_audio_preprocessor_int8_model_data);
  InjectAllocationFailures("micro_speech_quantized",
                           g_micro_speech_quantized_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, MnistLstmModels) {
  InjectAllocationFailures("trained_lstm", g_trained_lstm_model_data);
  InjectAllocationFailures("trained_lstm_int8", g_trained_lstm_int8_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, AddModels) {
  InjectAllocationFailures("simple_add_model", g_simple_add_model_model_data);
  InjectAllocationFailures("add_four_numbers", g_add_four_numbers_model_data);
}

TEST(HeliaArenaExhaustionInjectionTest, ResourceVariablesModel) {
  InjectAllocationFailures("resource_variables",
                           g_resource_variables_model_data,
                           /*resource_variables=*/true);
}

TF_LITE_MICRO_TESTS_MAIN
