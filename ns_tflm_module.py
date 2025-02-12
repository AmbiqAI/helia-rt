# # Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Starting point for writing scripts to integrate TFLM with external IDEs.

# This script can be used to output a tree containing only the sources and headers
# needed to use TFLM for a specific configuration (e.g. target and
# optimized_kernel_implementation). This should serve as a starting
# point to integrate TFLM with external IDEs.

# The goal is for this script to be an interface that is maintained by the TFLM
# team and any additional scripting needed for integration with a particular IDE
# should be written external to the TFLM repository and built to work on top of
# the output tree generated with this script.

# We will add more documentation for a desired end-to-end integration workflow as
# we get further along in our prototyping. See this github issue for more details:
#   https://github.com/tensorflow/tensorflow/issues/47413
# """

#!/usr/bin/env python3
import argparse
import fileinput
import os
import re
import shutil
import subprocess

def _get_dirs(file_list):
  dirs = set()
  for filepath in file_list:
    dirs.add(os.path.dirname(filepath))
  return dirs

def _get_file_list(key, makefile_options, tensorflow_root):
  params_list = [
      "make", "-f",
      tensorflow_root + "tensorflow/lite/micro/tools/make/Makefile", key
  ] + makefile_options.split()
  process = subprocess.Popen(params_list,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

  if process.returncode != 0:
    raise RuntimeError("%s failed with \n\n %s" %
                       (" ".join(params_list), stderr.decode()))

  return [bytepath.decode() for bytepath in stdout.split()]

def _third_party_src_and_dest_files(prefix_dir, makefile_options,
                                    tensorflow_root):
  src_files = []
  src_files.extend(
      _get_file_list("list_third_party_sources", makefile_options,
                     tensorflow_root))
  src_files.extend(
      _get_file_list("list_third_party_headers", makefile_options,
                     tensorflow_root))

  # The list_third_party_* rules give paths relative to the root of the git repo.
  # In the output tree, we want the code under prefix_dir/third_party.
  tflm_download_path = os.path.join(tensorflow_root,
                                    "tensorflow/lite/micro/tools/make/downloads")
  dest_files = []

  m_profile_dir = os.path.join(tflm_download_path, "cmsis/CMSIS/Core/Include/m-profile")
  # or whichever path you actually need
  m_profile_headers = []
  for root, _, files in os.walk(m_profile_dir):
      for file in files:
        full_path = os.path.join(root, file)
        m_profile_headers.append(full_path)
  src_files.extend(m_profile_headers)

  third_party_path = os.path.join(prefix_dir, "third_party")

  for f in src_files:
    if os.path.isabs(f):
      # If the submodule path was given as an absolute path
      dest_files.append(os.path.normpath(third_party_path + f))
    else:
      dest_files.append(
          os.path.join(third_party_path,
                       os.path.relpath(f, tflm_download_path)))

  return src_files, dest_files

def _tflm_src_and_dest_files(prefix_dir, makefile_options, tensorflow_root):
  src_files = []
  src_files.extend(
      _get_file_list("list_library_sources", makefile_options,
                     tensorflow_root))
  src_files.extend(
      _get_file_list("list_library_headers", makefile_options,
                     tensorflow_root))
  dest_files = [os.path.join(prefix_dir, src) for src in src_files]
  return src_files, dest_files

def _get_src_and_dest_files(prefix_dir, makefile_options, tensorflow_root):
  tflm_src_files, tflm_dest_files = _tflm_src_and_dest_files(
      prefix_dir, makefile_options, tensorflow_root)
  third_party_srcs, third_party_dests = _third_party_src_and_dest_files(
      prefix_dir, makefile_options, tensorflow_root)

  all_src_files = tflm_src_files + third_party_srcs
  all_dest_files = tflm_dest_files + third_party_dests
  return all_src_files, all_dest_files

def _copy(src_files, dest_files):
  """Copy from src_files[i] to dest_files[i]."""
  for dirname in _get_dirs(dest_files):
    os.makedirs(dirname, exist_ok=True)

  for src, dst in zip(src_files, dest_files):
    shutil.copy(src, dst)

def _sync_back(dest_files, src_files):
  """Copy from dest_files[i] back to src_files[i]."""
  for dirname in _get_dirs(src_files):
    # there is one dirname that is an emptry string due to the LICENSE file, we can ignore that one
    if dirname == '':
       continue
    os.makedirs(dirname, exist_ok=True)

  for dst, src in zip(dest_files, src_files):
    if os.path.isfile(dst):
      # Copy if the file exists in the output directory
      shutil.copy(dst, src)

def _get_tflm_generator_path(tensorflow_root):
  return _get_file_list("list_generator_dir",
                        "TENSORFLOW_ROOT=" + tensorflow_root,
                        tensorflow_root)[0]

def _create_examples_tree(prefix_dir, examples_list, tensorflow_root):
  files = []
  for e in examples_list:
    files.extend(
        _get_file_list("list_%s_example_sources" % (e),
                       "TENSORFLOW_ROOT=" + tensorflow_root, tensorflow_root))
    files.extend(
        _get_file_list("list_%s_example_headers" % (e),
                       "TENSORFLOW_ROOT=" + tensorflow_root, tensorflow_root))

  tflm_examples_path = os.path.join(tensorflow_root, "tensorflow/lite/micro/examples")
  tflm_downloads_path = os.path.join(tensorflow_root, "tensorflow/lite/micro/tools/make/downloads")
  tflm_generator_path = _get_tflm_generator_path(tensorflow_root)

  dest_file_list = []
  for f in files:
    if tflm_generator_path in f:
      relative_path = os.path.relpath(f, tflm_generator_path)
      full_filename = os.path.join(prefix_dir, relative_path)
    elif tflm_examples_path in f:
      relative_path = os.path.relpath(f, tflm_examples_path)
      full_filename = os.path.join(prefix_dir, "examples", relative_path)
    elif tflm_downloads_path in f:
      relative_path = os.path.relpath(f, tflm_downloads_path)
      full_filename = os.path.join(prefix_dir, "third_party", relative_path)
    else:
      full_filename = os.path.join(prefix_dir, f)
    dest_file_list.append(full_filename)

  for dest_file, filepath in zip(dest_file_list, files):
    dest_dir = os.path.dirname(dest_file)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(filepath, dest_file)

  # Since we are changing the directory structure for the examples, we must
  # update include paths in the code to reflect the new structure.
  tflm_examples_include_path = "tensorflow/lite/micro/examples"
  examples_gen_include_path = os.path.join(tensorflow_root, "tensorflow/lite/micro/examples")
  for filepath in dest_file_list:
    with fileinput.FileInput(filepath, inplace=True) as f:
      for line in f:
        include_match = re.match(
            r'.*#include.*"' + tflm_examples_include_path + r'/([^/]+)/.*"',
            line)
        examples_gen_include_match = re.match(
            r'.*#include.*"' + examples_gen_include_path + r'/([^/]+)/.*"',
            line)
        if include_match:
          text_to_replace = os.path.join(tflm_examples_include_path,
                                         include_match.group(1)) + "/"
          line = line.replace(text_to_replace, "")
        elif examples_gen_include_match:
          text_to_replace_1 = os.path.join(
              examples_gen_include_path,
              examples_gen_include_match.group(1)) + "/"
          line = line.replace(text_to_replace_1, "")
        print(line, end="")

def _rename_cc_to_cpp(output_dir):
  for path, _, files in os.walk(output_dir):
    for name in files:
      if name.endswith(".cc"):
        base_name_with_path = os.path.join(path, os.path.splitext(name)[0])
        os.rename(base_name_with_path + ".cc", base_name_with_path + ".cpp")

def _generate_module_mk(
    output_dir,
    dest_files,
    target_arch="cortex-m55",
    optimized_kernel_dir="CMSIS_NN"
):
    module_mk_path = os.path.join(output_dir, "module.mk")
    source_extensions = (".c", ".cc", ".cpp")
    source_files = []
    for df in dest_files:
        if df.endswith(source_extensions):
            # Make path relative to output_dir for easier references
            rel_path = os.path.relpath(df, output_dir)
            source_files.append(rel_path)

    # Filter out any 'third_party' or test files
    filtered_srcs = [sf for sf in source_files
                     if ("third_party" not in sf and "test" not in sf)]

    with open(module_mk_path, "w") as f:
        f.write(f"TARGET_ARCH := {target_arch}\n\n")
        f.write("CORE_OPTIMIZATION_LEVEL     := -Os\n")
        f.write("KERNEL_OPTIMIZATION_LEVEL   := -O2\n")
        f.write("COMMON_FLAGS := \\\n")
        f.write("  -Wall -Wextra -Wno-unused-parameter \\\n")
        f.write("  -Wsign-compare -Wdouble-promotion -Wunused-variable -Wswitch -Wvla \\\n")
        f.write("  -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 \\\n")
        f.write("  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON\n\n")

        f.write("CXXFLAGS += -std=c++17 -fno-rtti -fno-exceptions $(COMMON_FLAGS)\n")
        f.write("CFLAGS   += $(COMMON_FLAGS)\n\n")

        f.write("# Set optimized kernel folder name:\n")
        f.write(f"OPTIMIZED_KERNEL_DIR := {optimized_kernel_dir}\n\n")
        f.write("ifneq ($(OPTIMIZED_KERNEL_DIR),)\n")
        f.write("\tADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])\n")
        f.write("endif\n\n")

        f.write("# Add ADDITIONAL_DEFINES to CFLAGS and CXXFLAGS\n")
        f.write("CFLAGS   += $(ADDITIONAL_DEFINES)\n")
        f.write("CXXFLAGS += $(ADDITIONAL_DEFINES)\n\n")

        f.write("# Include paths.\n")
        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/$(OPTIMIZED_KERNEL_DIR)\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/$(OPTIMIZED_KERNEL_DIR)\n\n")

        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/flatbuffers/include\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/flatbuffers/include\n\n")

        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/gemmlowp\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/gemmlowp\n\n")

        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/kissfft\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/kissfft\n\n")

        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/ruy\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/ruy\n\n")

        f.write("CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/cmsis/CMSIS/Core/Include\n")
        f.write("CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/cmsis/CMSIS/Core/Include\n\n")

        f.write("# Paths to precompiled third-party libraries.\n")
        f.write("lib_prebuilt += ns-tflm/treedir/libtensorflow-microlite.a\n\n")

        f.write("# Dynamic collection of TFLM source files (skips 'third_party').\n")
        f.write("local_src := \\\n")
        for sf in filtered_srcs:
            f.write("ns-tflm/treedir/" + f"{sf} \\\n")
        f.write("\n")

        f.write("# Include the Cortex M generic makefile\n")
        f.write("include ns-tflm/treedir/cortex_m_generic_makefile.inc\n\n")

        f.write("$(eval $(call make-library, $(local_bin)/ns-tflm.a, $(local_src)))\n")

    print(f"Generated {module_mk_path} with {len(filtered_srcs)} files.")

def main():
    parser = argparse.ArgumentParser(
        description="Starting script for TFLM project generation"
    )
    parser.add_argument(
        "action",
        choices=["create", "sync"],
        help=(
            "create = generate output directory and module.mk.\n"
            "sync   = copy files from output_dir back to original locations."
        )
    )
    parser.add_argument("output_dir", help="Output directory for generated TFLM tree (for create) or existing TFLM tree (for sync).")
    parser.add_argument("--no_copy",
                        action="store_true",
                        help="Do not copy files to output directory when action=create.")
    parser.add_argument("--print_src_files",
                        action="store_true",
                        help="Print the list of TFLM source files.")
    parser.add_argument(
        "--print_dest_files",
        action="store_true",
        help="Print the list of destination files in output_dir.")
    parser.add_argument(
        "--makefile_options",
        default="",
        help=(
            "Additional TFLM Makefile options. E.g.:\n"
            '--makefile_options="TARGET=<target> TENSORFLOW_ROOT=<root> OPTIMIZED_KERNEL_DIR=<dir> TARGET_ARCH=cortex-m4"'
        )
    )
    parser.add_argument(
        "--examples", "-e",
        action="append",
        help=(
            "Examples to add to the output tree.\n"
            "Usage: -e hello_world -e micro_speech"
        )
    )
    parser.add_argument(
        "--rename_cc_to_cpp",
        action="store_true",
        help="Rename all .cc files to .cpp in the destination."
    )

    args = parser.parse_args()

    # Parse makefile options
    make_entries = args.makefile_options.split()
    tensorflow_root = ""
    target_arch = ""
    optimized_kernel_dir = ""
    for make_entry in make_entries:
        key_value = make_entry.split("=")
        if len(key_value) == 2:
            key, val = key_value
            if key == "TENSORFLOW_ROOT":
                tensorflow_root = val
            elif key == "TARGET_ARCH":
                target_arch = val
            elif key == "OPTIMIZED_KERNEL_DIR":
                optimized_kernel_dir = val
    if tensorflow_root == "":
       tensorflow_root = "./"

    # We'll always need to figure out the source/dest lists
    src_files, dest_files = _get_src_and_dest_files(
        args.output_dir,
        args.makefile_options,
        tensorflow_root
    )

    if args.action == "create":
        # Ensure the third-party downloads are present
        params_list = [
            "make", "-f", os.path.join(tensorflow_root,
                                       "tensorflow/lite/micro/tools/make/Makefile"),
            "third_party_downloads"
        ] + make_entries
        process = subprocess.Popen(params_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("%s failed with \n\n %s" %
                               (" ".join(params_list), stderr.decode()))

        if args.print_src_files:
            print(" ".join(src_files))
        if args.print_dest_files:
            print(" ".join(dest_files))

        # Copy src files -> output_dir, unless user specified --no_copy
        if not args.no_copy:
            _copy(src_files, dest_files)

        # Create examples if requested
        if args.examples is not None:
            _create_examples_tree(args.output_dir, args.examples, tensorflow_root)

        # Rename .cc -> .cpp if requested
        if args.rename_cc_to_cpp:
            _rename_cc_to_cpp(args.output_dir)

        # Copy a custom cortex_m_generic_makefile.inc
        makefile_inc_src = os.path.join(tensorflow_root, "neuralspot/cortex_m_generic_makefile.inc")
        makefile_inc_dst = os.path.join(args.output_dir, "cortex_m_generic_makefile.inc")
        os.makedirs(os.path.dirname(makefile_inc_dst), exist_ok=True)
        shutil.copy(makefile_inc_src, makefile_inc_dst)

        # Copy ARMCM55.h, system_ARMCM55.h
        armcm55_h_src = os.path.join(
            tensorflow_root,
            "tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM55/Include/ARMCM55.h"
        )
        armcm55_h_dst = os.path.join(args.output_dir, "third_party/cmsis/CMSIS/Core/Include/ARMCM55.h")
        os.makedirs(os.path.dirname(armcm55_h_dst), exist_ok=True)
        shutil.copy(armcm55_h_src, armcm55_h_dst)
        dest_files.append(armcm55_h_dst)

        system_cm55_h_src = os.path.join(
            tensorflow_root,
            "tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM55/Include/system_ARMCM55.h"
        )
        system_cm55_h_dst = os.path.join(args.output_dir, "third_party/cmsis/CMSIS/Core/Include/system_ARMCM55.h")
        os.makedirs(os.path.dirname(system_cm55_h_dst), exist_ok=True)
        shutil.copy(system_cm55_h_src, system_cm55_h_dst)
        dest_files.append(system_cm55_h_dst)

        # Copy ARMCM4.h, system_ARMCM4.h
        armcm4_h_src = os.path.join(
            tensorflow_root,
            "tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM4/Include/ARMCM4.h"
        )
        armcm4_h_dst = os.path.join(args.output_dir, "third_party/cmsis/CMSIS/Core/Include/ARMCM4.h")
        os.makedirs(os.path.dirname(armcm4_h_dst), exist_ok=True)
        shutil.copy(armcm4_h_src, armcm4_h_dst)
        dest_files.append(armcm4_h_dst)

        system_cm4_h_src = os.path.join(
            tensorflow_root,
            "tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM4/Include/system_ARMCM4.h"
        )
        system_cm4_h_dst = os.path.join(args.output_dir, "third_party/cmsis/CMSIS/Core/Include/system_ARMCM4.h")
        os.makedirs(os.path.dirname(system_cm4_h_dst), exist_ok=True)
        shutil.copy(system_cm4_h_src, system_cm4_h_dst)
        dest_files.append(system_cm4_h_dst)

        # Copy libtensorflow-microlite.a
        lib_prebuilt_src = os.path.join(tensorflow_root, "neuralspot/libtensorflow-microlite.a")
        lib_prebuilt_dst = os.path.join(args.output_dir, "libtensorflow-microlite.a")
        os.makedirs(os.path.dirname(lib_prebuilt_dst), exist_ok=True)
        shutil.copy(lib_prebuilt_src, lib_prebuilt_dst)
        dest_files.append(lib_prebuilt_dst)

        # Copy xtensa headers, needed by Ambiq pad.cc
        xtensa_pad_src = os.path.join(tensorflow_root,
                                      "tensorflow/lite/micro/kernels/xtensa/xtensa_pad.h")
        xtensa_h_src = os.path.join(tensorflow_root,
                                    "tensorflow/lite/micro/kernels/xtensa/xtensa.h")

        xtensa_pad_dst = os.path.join(args.output_dir,
                                      "tensorflow/lite/micro/kernels/xtensa/xtensa_pad.h")
        xtensa_h_dst = os.path.join(args.output_dir,
                                    "tensorflow/lite/micro/kernels/xtensa/xtensa.h")

        os.makedirs(os.path.dirname(xtensa_pad_dst), exist_ok=True)
        os.makedirs(os.path.dirname(xtensa_h_dst), exist_ok=True)

        shutil.copy(xtensa_pad_src, xtensa_pad_dst)
        shutil.copy(xtensa_h_src, xtensa_h_dst)
        dest_files.append(xtensa_pad_dst)
        dest_files.append(xtensa_h_dst)

        # Finally, generate the module.mk
        _generate_module_mk(
            args.output_dir,
            dest_files,
            target_arch=target_arch,
            optimized_kernel_dir=optimized_kernel_dir
        )

    elif args.action == "sync":
        # In "sync" mode, we do the reverse copy: output_dir -> original
        # This allows copying back any changes made in the output tree.
        if args.print_src_files:
            print(" ".join(src_files))
        if args.print_dest_files:
            print(" ".join(dest_files))

        print("[SYNC MODE] Copying files from output_dir back to original locations...")
        _sync_back(dest_files, src_files)
        print("[SYNC MODE] Done. Files have been copied back to their original paths.")

if __name__ == "__main__":
    main()
