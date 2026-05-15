# Minimal Zephyr application that links the heliaRT library and calls into
# its public API. Used by the smoke_cmake workflow's zephyr-build job.

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_zephyr_smoke LANGUAGES C CXX)

target_sources(app PRIVATE src/main.cc)
