if(CONFIG_HELIA_RT)

  set(HELIA_RT_DIR ${ZEPHYR_CURRENT_MODULE_DIR})

  zephyr_library_named(helia_rt)

  zephyr_include_directories(
    ${HELIA_RT_DIR}/.
    ${HELIA_RT_DIR}/third_party/gemmlowp
    ${HELIA_RT_DIR}/third_party/flatbuffers/include
    ${HELIA_RT_DIR}/third_party/ruy
    ${HELIA_RT_DIR}/third_party/kissfft
    ${HELIA_RT_DIR}/third_party/cmsis_nn/Include
  )

  zephyr_library_compile_definitions(NS_CMSIS_NN)
  zephyr_library_compile_definitions(AMBIQ)

  if(CONFIG_CPU_CORTEX_M55)
    set(HELIA_RT_ARCH_TOKEN cm55)
  elseif(CONFIG_CPU_CORTEX_M4 AND CONFIG_FPU)
    set(HELIA_RT_ARCH_TOKEN cm4)
  else()
    message(FATAL_ERROR
      "Unsupported heliaRT prebuilt arch. Supported Zephyr prebuilts are Cortex-M4 with FPU and Cortex-M55.")
  endif()

  if(ZEPHYR_TOOLCHAIN_VARIANT STREQUAL "armclang")
    set(HELIA_RT_TOOLCHAIN_TOKEN armclang)
  elseif(
    ZEPHYR_TOOLCHAIN_VARIANT STREQUAL "gnuarmemb" OR
    ZEPHYR_TOOLCHAIN_VARIANT STREQUAL "zephyr" OR
    ZEPHYR_TOOLCHAIN_VARIANT STREQUAL "cross-compile")
    set(HELIA_RT_TOOLCHAIN_TOKEN gcc)
  else()
    message(FATAL_ERROR
      "Unsupported heliaRT prebuilt toolchain '${ZEPHYR_TOOLCHAIN_VARIANT}'. Supported toolchains are gcc-compatible and armclang.")
  endif()

  if(CONFIG_HELIA_RT_PREBUILT_BUILD_DEBUG)
    set(HELIA_RT_BUILD_TOKEN debug)
  elseif(CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE_WITH_LOGS)
    set(HELIA_RT_BUILD_TOKEN release-with-logs)
  else()
    set(HELIA_RT_BUILD_TOKEN release)
  endif()

  set(HELIA_RT_PREBUILT_LIB
    ${HELIA_RT_DIR}/lib/libhelia-rt-zephyr-${HELIA_RT_ARCH_TOKEN}-${HELIA_RT_TOOLCHAIN_TOKEN}-${HELIA_RT_BUILD_TOKEN}.a)

  if(NOT EXISTS "${HELIA_RT_PREBUILT_LIB}")
    message(FATAL_ERROR
      "Missing heliaRT prebuilt archive '${HELIA_RT_PREBUILT_LIB}'. Confirm that the selected arch, toolchain, and build flavor are part of the published bundle matrix.")
  endif()

  add_library(helia_rt_prebuilt STATIC IMPORTED GLOBAL)
  set_target_properties(helia_rt_prebuilt PROPERTIES IMPORTED_LOCATION "${HELIA_RT_PREBUILT_LIB}")
  target_link_libraries(helia_rt PUBLIC helia_rt_prebuilt)

endif()
