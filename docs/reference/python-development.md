# Python Development

heliaRT keeps the upstream LiteRT for Micro Python utilities available for development, testing, and model inspection. Most repository Python workflows run through Bazel so dependency setup stays reproducible in CI.


* [TensorFlow Python style guide](https://www.tensorflow.org/community/contribute/code_style#python_style)


## Using Bazel

We use Bazel as our default build system for Python and the continuous
integration infrastructure runs Python unit tests through Bazel.

When using Bazel with Python, all the environment setup is handled as part of the
build.

Some example commands:
```sh
bazel test tensorflow/lite/tools:flatbuffer_utils_test
bazel build tensorflow/lite/tools:visualize

bazel-bin/tensorflow/lite/tools/visualize tensorflow/lite/micro/models/person_detect.tflite tensorflow/lite/micro/models/person_detect.tflite.html
```

## Manual Setup Illustration

For advanced users that would like to use the Python code in the LiteRT for Micro repository
independent of Bazel, here is one approach.

Please note that this setup is unsupported and will need users to debug various
issues on their own. It is described here for illustrative purposes only.

```sh
# The cloned helia-rt folder needs to be renamed to helia_rt
mv helia-rt helia_rt
# To set up a specific Python version, make sure `python` is pointed to the
# desired version. For example, call `python3.11 -m venv helia_rt/venv`.
python -m venv helia_rt/venv
echo "export PYTHONPATH=\${PYTHONPATH}:${PWD}" >> helia_rt/venv/bin/activate
cd helia_rt
source venv/bin/activate
pip install --upgrade pip
pip install -r third_party/python_requirements.txt

# (Optional)
pip install ipython
```

Run some tests and binaries:
```sh
python tensorflow/lite/tools/flatbuffer_utils_test.py
python tensorflow/lite/tools/visualize.py tensorflow/lite/micro/models/person_detect.tflite tensorflow/lite/micro/models/person_detect.tflite.html
```
