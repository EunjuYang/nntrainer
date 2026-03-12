---
title: Getting Started
...

# Getting Started

## Prerequisites

The following dependencies are needed to compile/build/run.

* gcc/g++ >= 7 ( std=c++17 is used )
   - (note)  >= 13 is recommended to enable fp16 support
* meson >= 0.55.0
* libopenblas-dev and base
* tensorflow-lite >= 2.3.0
* libiniparser
* libjsoncpp >=0.6.0 ( if you want to use open AI )
* libcurl3 >=7.47 ( if you want to use open AI )
* libgtest >=1.10 ( for testing )

## Install via PPA repository (Debian/Ubuntu)

The NNTrainer releases are available with launchpad PPA repository. In order to install it, use:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt update
sudo apt install nntrainer
```

## Clean build with pdebuild (Ubuntu 18.04)

Use the NNStreamer PPA to resolve additional build-dependencies (Tensorflow/Tensorflow-Lite).

Install build tools:

```bash
sudo apt install pbuilder debootstrap devscripts
```

The following example configuration is for Ubuntu 18.04:

You can configure the pbuilderrc file as follows:
```bash
$ cat ~/.pbuilderrc

DISTRIBUTION=bionic
COMPONENTS="main restricted universe multiverse"
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION} main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION}-security main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION}-updates main restricted universe multiverse |\
  deb [trusted=yes] http://ppa.launchpad.net/nnstreamer/ppa/ubuntu ${DISTRIBUTION} main"
```

Then You can Link to ```/root/.pbuilderrc``` and create
```bash
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc

$ sudo pbuilder create
```

Run pdebuild to build and get the package.

```bash
$ pdebuild
...
$ ls -al /var/cache/pbuilder/result/*.deb
```

Refer to [PbuilderHowto](https://wiki.ubuntu.com/PbuilderHowto) for more about pdebuild.

## Linux Self-Hosted Build

### Build with Debian/Ubuntu tools

#### Clone the needed repositories

```bash
git clone https://github.com/nntrainer/nntrainer
```

Alternatively, you may simply download binary packages from PPA:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt install tensorflow2-lite-dev
```

#### Build .deb package

```bash
cd nntrainer && sudo mk-build-deps --install debian/control && sudo dpkg -i *.deb
```

#### Creating the .deb packages

```bash
export DEB_BUILD_OPTIONS="parallel=$(($(cat /proc/cpuinfo |grep processor|wc -l) + 1))"
cd nntrainer && time debuild -us -uc
```

If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add `-uc -us` options to `debuild`.

#### Install the generated \*.deb files

The files will be there at the parent dir. Eg. at `nnbuilder/..` directory.

In order to install them (should run as root):

```bash
sudo apt install ./tensorflow-lite-dev_*.deb
sudo apt install ./nntrainer_0.1.0*_amd64.deb
```

If you need nntrainer development package:

```bash
sudo apt install ./nntrainer-dev_0.1.0*_amd64.deb
```

### Build with meson

Add nnstreamer ppa for some of the dependencies.

```bash
sudo add-apt-repository ppa:nnstreamer/ppa
sudo apt-get update
```

Install the required packages.

```bash
sudo apt install meson ninja-build
sudo apt install gcc g++ pkg-config libopenblas-dev libiniparser-dev libjsoncpp-dev libcurl3-dev tensorflow2-lite-dev nnstreamer-dev libglib2.0-dev libgstreamer1.0-dev libgtest-dev ml-api-common-dev flatbuffers-compiler ml-inference-api-dev
```

Install submodules.
```bash
git submodule sync && git submodule update --init --depth 1
```

Build at the git repo root directory, this will install nntrainer and related files.

```bash
meson build
ninja -C build install
```

* Installs libraries to ```{prefix}/{libdir}```
* Installs common header files to ```{prefix}/{includedir}```

## Build with Transformer (LLM) Support

NNTrainer supports transformer-based LLM inference (LLaMA, PicoGPT, CausalLM) via the `enable-transformer` option.

### Prerequisites

In addition to the base prerequisites, the following packages are required:

```bash
sudo apt install meson ninja-build gcc g++ pkg-config cmake
sudo apt install libopenblas-dev flatbuffers-compiler libjsoncpp-dev libcurl4-openssl-dev
```

### Initialize submodules

Submodules (OpenBLAS, iniparser, googletest, etc.) must be initialized before building:

```bash
git submodule sync && git submodule update --init --depth 1
```

### Build (core library only)

To build just the core nntrainer library with transformer support:

```bash
meson setup build -Denable-transformer=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-app=false -Denable-test=false
ninja -C build
```

### Build (with transformer applications)

To include transformer applications (LLaMA, PicoGPT, CausalLM):

```bash
meson setup build -Denable-transformer=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-test=false
ninja -C build
```

### Build (recommended for PC testing with FP16)

```bash
meson setup build -Denable-transformer=true -Denable-fp16=true -Dthread-backend=omp -Domp-num-threads=4 -Denable-tflite-backbone=false -Denable-tflite-interpreter=false
ninja -C build
```

### Important notes

* `enable-transformer=true` automatically disables `mmap-read` (they are incompatible).
* If `libopenblas-dev` is not installed as a system package, the build system will attempt to compile OpenBLAS from the subproject source. This may fail on environments without AVX512 support. Installing the system package (`sudo apt install libopenblas-dev`) is recommended.
* TFLite backbone/interpreter can be disabled if `tensorflow2-lite` is not available in your environment.
* For CausalLM application details, see [Applications/CausalLM/README.md](../Applications/CausalLM/README.md).

## Build on Tizen

Get GBS from <https://docs.tizen.org/platform/developing/building/>

First install the required packages.

```bash
sudo apt install gbs
```

Generates .rpm packages:

```bash
gbs build
```

`gbs build` will execute unit testing as well unlike meson build.

## Troubleshooting

### Error 1:

```bash
In file included from /usr/include/tensorflow/lite/core/api/op_resolver.h:20,
                 from /usr/include/tensorflow/lite/model.h:39,
                 from /usr/include/tensorflow/lite/kernels/register.h:19,
                 from ../Applications/KNN/jni/main_sample.cpp:21:
/usr/include/tensorflow/lite/schema/schema_generated.h:21:10: fatal error: flatbuffers/flatbuffers.h: No such file or directory
   21 | #include "flatbuffers/flatbuffers.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

### Solution: Please install libflatbuffers-dev using the following:

```bash
sudo apt install libflatbuffers-dev
```

## For Windows OS please follow: [Getting Started (Windows)](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started-windows.md)
