# This tool builds all the TFLM versions needed for neuralSPOT,  creates the treedir, and copies the whole thing to neuralSPOTs extern/tensorflow directory

import os
import pydantic_argparse
from pydantic import BaseModel, Field
import subprocess

class Params(BaseModel):
    copy_tflm: bool = Field(True, description="Copy the files to the neuralSPOT directory",)
    build_tflm: bool = Field(True, description="Build the TFLM versions",)
    treedir: bool = Field(False, description="Create the treedir",)
    destdir: str = Field("../ns-mirror/extern/tensorflow", description="Destination directory for the files")
    tflm_name: str = Field("auto", description="Name of the tensorflow lite micro directory, auto will use the current date")

def create_parser():
    return pydantic_argparse.ArgumentParser(
        model=Params,
        prog="This tool builds all the TFLM versions needed for neuralSPOT,  creates the treedir, and copies the whole thing to neuralSPOTs extern/tensorflow directory",
        description="Compile and install TFLM versions for neuralSPOT",
    )

def main():
    # parser = create_parser()
    # print (parser)

    # params = parser.parse_typed_args()
    # print(params)
    
    release_types = ["debug", "release", "release_with_logs"]
    ns_release_names = ["debug", "release", "release-with-logs"]
    compilers = ["gcc", "armclang"]
    processors = ["cortex-m4+fp", "cortex-m55"]

    # if params.build_tflm:
         # Build 3 TFLM release types for each compiler and each target processor
    for compiler in compilers:
        for processor in processors:
            co_processor_flag = ""
            if processor == "cortex-m55":
                co_processor_flag = "ambiq"
            for release_type in release_types:
                print(f"Building {compiler} {processor} {co_processor_flag} {release_type}")
                print(f"make -f ./tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH={processor} TOOLCHAIN={compiler} OPTIMIZED_KERNEL_DIR=cmsis_nn  CO_PROCESSOR={co_processor_flag} BUILD_TYPE={release_type} microlite")
                os.system(f"make -f ./tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH={processor} TOOLCHAIN={compiler} OPTIMIZED_KERNEL_DIR=cmsis_nn CO_PROCESSOR={co_processor_flag} BUILD_TYPE={release_type} microlite -j8")
            
    # create the treedir
    # if params.treedir:
    print ("Creating treedir")
    print ("python3 ./tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py --makefile_options \"TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn CO_PROCESSOR=ambiq\" treedir")
    os.system('python3 ./tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py --makefile_options "TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn CO_PROCESSOR=ambiq" treedir')

    # delete all .c and .cc files in treedir
    print ("Deleting all .c and .cc files in treedir")
    os.system("find treedir -type f -name '*.c' -delete")
    os.system("find treedir -type f -name '*.cc' -delete")

    # Create a directory over in ns-mirror/extern/tensorflow and copy the files over
    # if params.copy_tflm:
    print ("Copying files to neuralSPOT")

    # if params.tflm_name == "auto":
    tflm_name = subprocess.check_output("date +'%Y_%m_%d'", shell=True).decode('utf-8').strip()
    tflm_name = "ns_tflm_" + str(tflm_name)
    tflm_path = os.path.join("../ns-mirror/extern/tensorflow", tflm_name)
    # make the directory, don't fail if it already exists
    os.makedirs(tflm_path, exist_ok=True)
    os.makedirs(os.path.join(tflm_path, "lib"), exist_ok=True)

    # copy all the static libs we just built to the lib directory
    for compiler in compilers:
        for processor in processors:
            for release_type in release_types:
                # convert processor type to the one used in the lib name
                if processor == "cortex-m4+fp":
                    ns_processor = "cm4"
                    label = "m4+fp"
                    co_processor_str = ""  # no CO_PROCESSOR for M4
                elif processor == "cortex-m55":
                    ns_processor = "cm55"
                    label = "m55"
                    co_processor_str = 'ambiq_' # inject 'ambiq' in the build folder name
                
                # replace _ with - in release type
                ns_release_type = release_type.replace("_", "-")

                dest_lib_name = f"libtensorflow-microlite-{ns_processor}-{compiler}-{co_processor_str}{ns_release_type}.a"
                source_lib = f"gen/cortex_m_generic_{processor}_{release_type}_cmsis_nn_{co_processor_str}{compiler}/lib/libtensorflow-microlite.a"
                print(f"Copying {source_lib} to {os.path.join(tflm_path, 'lib', dest_lib_name)}")
                os.system(f"cp {source_lib} {os.path.join(tflm_path, 'lib', dest_lib_name)}")

    # os.makedirs(os.path.join(tflm_path, "signal"))
    # os.makedirs(os.path.join(tflm_path, "tensorflow"))
    # os.makedirs(os.path.join(tflm_path, "third_party"))

    # copy directories from treedir to tflm_path subdirectories
    os.system(f"cp -r treedir/* {tflm_path}")
    
    print (f"Copying to {tflm_path}")
    os.system(f"cp -r treedir/* {tflm_path}")
    print (tflm_path)



if __name__ == "__main__":
    main()