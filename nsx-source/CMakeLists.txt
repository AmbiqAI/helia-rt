# Source-provider module for consumers that compile helia-rt themselves.
#
# It intentionally contributes no runtime sources. Its presence causes NSX to
# vendor this project so sibling modules can use the canonical helia-rt source
# tree without also instantiating the nsx::helia_rt runtime module.

cmake_minimum_required(VERSION 3.21)

add_library(helia_rt_source INTERFACE)
add_library(helia_rt::source ALIAS helia_rt_source)
