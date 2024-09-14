
<!-- omit in toc -->
# Getting Started

In this document, we explain how to install the NNTrainer. We provide four installation methods. Please begin by reviewing the prerequisites, then choose the method that best fits your environment. For contributors, it is highly recommended to follow the instructions in the [`Build with meson`](#3-2-build-with-meson) section to build the project.

- [Step 0) Prerequisites](#step-0-prerequisites)
- [Step 1) Build and Install](#step-1-build-and-install)
  - [Build on Linux (Debian / Ubuntu)](#build-on-linux-debian--ubuntu)
    - [(1) Install via PPA repository (Debian/Ubuntu)](#1-install-via-ppa-repository-debianubuntu)
    - [(2) Clean build with pdebuild (Ubuntu 18.04)](#2-clean-build-with-pdebuild-ubuntu-1804)
    - [(3) Linux Self-Hosted Build](#3-linux-self-hosted-build)
      - [(3-1) Build with Debian/Ubuntu tools](#3-1-build-with-debianubuntu-tools)
      - [(3-2) Build with meson](#3-2-build-with-meson)
  - [Build on Tizen](#build-on-tizen)
- [Troubleshooting](#troubleshooting)


## Step 0) Prerequisites

The following dependencies are needed to compile/build/run.

* gcc/g++ >= 7 ( std=c++17 is used )
* meson >= 0.50
* libopenblas-dev and base
* tensorflow-lite >= 2.3.0
* libiniparser
* libjsoncpp >=0.6.0 ( if you want to use open AI )
* libcurl3 >=7.47 ( if you want to use open AI )
* libgtest >=1.10 ( for testing )

## Step 1) Build and Install

### Build on Linux (Debian / Ubuntu)
#### (1) Install via PPA repository (Debian/Ubuntu)

The NNTrainer releases are available with launchpad PPA repository. In order to install it, use:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt update
sudo apt install nntrainer
```

---

#### (2) Clean build with pdebuild (Ubuntu 18.04)

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

---

#### (3) Linux Self-Hosted Build

##### (3-1) Build with Debian/Ubuntu tools

<!-- omit in toc -->
###### Clone the needed repositories

```bash
git clone https://github.com/nnstreamer/nntrainer
```

Alternatively, you may simply download binary packages from PPA:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt install tensorflow2-lite-dev
```

<!-- omit in toc -->
###### Build .deb package

```bash
cd nntrainer && sudo mk-build-deps --install debian/control && sudo dpkg -i *.deb
```

<!-- omit in toc -->
###### Creating the .deb packages

```bash
export DEB_BUILD_OPTIONS="parallel=$(($(cat /proc/cpuinfo |grep processor|wc -l) + 1))"
cd nntrainer && time debuild -us -uc
```

If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add `-uc -us` options to `debuild`.

<!-- omit in toc -->
###### Install the generated \*.deb files

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

##### (3-2) Build with meson

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

Build at the git repo root directory, this will install nntrainer and related files.

```bash
meson build
ninja -C build install
```

* Installs libraries to ```{prefix}/{libdir}```
* Installs common header files to ```{prefix}/{includedir}```

### Build on Tizen

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

<!-- omit in toc -->
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

<!-- omit in toc -->
### Solution: Please install libflatbuffers-dev using the following:

```bash
sudo apt install libflatbuffers-dev
```
