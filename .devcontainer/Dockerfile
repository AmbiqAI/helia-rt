# This docker container can be used to run all the TFLM CI checks.
#
# It is only used as part of the GitHub workflows to test for code-style. But
# the container is available and ready for use to run all the checks locally,
# in case that is useful for debugging. See all the versions at
# https://github.com/users/TFLM-bot/packages/container/tflm-ci/versions
#
# docker pull ghcr.io/tflm-bot/tflm-ci:<version>
#
# Build you own container with:
# docker build -f ci/Dockerfile.micro -t tflm-ci .
#
# Use a prebuilt Python image instead of base Ubuntu to speed up the build process,
# since it has all the build dependencies we need for Micro and downloads much faster
# than the install process.

# Using a multistage build so that the build tools required for stage 1 don't make the
# CI container unnecessarily large.
#
# Base image: Debian 12 "bookworm" (glibc 2.36, libstdc++ from GCC 12). Required by
# Arm Toolchain for Embedded (ATfE) 22.1.0 prebuilt binaries, which need GLIBC_2.34
# and GLIBCXX_3.4.30 / CXXABI_1.3.13. Bullseye (glibc 2.31, libstdc++ from GCC 10)
# is too old: clang/clang++ load but every compile aborts with
#   `version 'GLIBC_2.34' not found` / `version 'GLIBCXX_3.4.30' not found`.
FROM python:3.10-bookworm AS qemu_builder
RUN apt-get update

RUN apt-get install -y ninja-build wget xz-utils
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev

COPY ci/install_qemu.sh ./
# Installs all built files into /qemu_install rather than /usr/local default.
RUN ./install_qemu.sh /tmp/qemu_install

# This stage is the final CI container.
# See note above on why bookworm (not bullseye) is required for ATfE 22.1.0.
FROM python:3.10-bookworm as tflm-ci

RUN apt-get update

RUN apt-get install -y zip xxd sudo

RUN apt install -y lsb-release wget software-properties-common gnupg
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16
RUN ln -s /usr/bin/clang-16 /usr/bin/clang
RUN ln -s /usr/bin/clang++-16 /usr/bin/clang++

RUN apt-get install -y clang-format-16
RUN ln -s /usr/bin/clang-format-16 /usr/bin/clang-format

# Needed when using the Dockerfile locally.
RUN git config --global --add safe.directory /opt/tflm

# Needed when the docker container is used with GitHub actions.
RUN git config --global --add safe.directory /github/workspace

# Install yapf to check for Python formatting as part of the TFLM continuous
# integration.
RUN pip install yapf==0.32.0

# Pillow was added first for the C array generation as a result of the following
# PRs:
# https://github.com/tensorflow/tflite-micro/pull/337
# https://github.com/tensorflow/tflite-micro/pull/410
RUN pip install Pillow

# necessary bits for create_size_log scripts
RUN pip install pandas
RUN pip install matplotlib
RUN pip install six

# Install Renode test dependencies
RUN pip install pyyaml requests psutil robotframework==4.0.1

# Install QEMU from build container qemu_builder into tflm-ci container.
# We're using a two stage build to keep the CI container smaller.
WORKDIR /usr/local
# Merge built files into /usr/local so that the path is already setup.
COPY --from=qemu_builder /tmp/qemu_install/. .

WORKDIR /

COPY ci/*.sh /install/

RUN /install/install_bazelisk.sh

RUN /install/install_buildifier.sh

# Install Arm Toolchain for Embedded (ATfE) — LLVM/Clang based bare-metal toolchain.
# Pinned to 22.1.0 (LLVM 22.1.0). Lives at /opt/atfe/bin which scripts auto-detect.
ARG ATFE_VERSION=22.1.0
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64)  ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Linux-x86_64.tar.xz" ;; \
      aarch64) ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Linux-AArch64.tar.xz" ;; \
      *) echo "Unsupported arch for ATfE: $(uname -m)"; exit 1 ;; \
    esac; \
    mkdir -p /opt/atfe; \
    wget -q "https://github.com/arm/arm-toolchain/releases/download/release-${ATFE_VERSION}-ATfE/${ATFE_ARCHIVE}" -O /tmp/atfe.tar.xz; \
    tar -xJf /tmp/atfe.tar.xz -C /opt/atfe --strip-components=1; \
    rm -f /tmp/atfe.tar.xz; \
    /opt/atfe/bin/clang --version
ENV PATH=/opt/atfe/bin:${PATH}
ENV ATFE_PREINSTALL=/opt/atfe

# Install uv (Python package manager) directly (no nanolayer dependency).
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Add a non-root user 'ambiqai' with passwordless sudo access
RUN useradd -m -s /bin/bash -G sudo ambiqai && \
    echo "ambiqai ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ambiqai && \
    chmod 0440 /etc/sudoers.d/ambiqai
