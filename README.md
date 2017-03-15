# Code Acceleration

The goal of this project is two fold. First, accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using using CUDA and OpenCL. Second, evaluate its performance on NVIDIA, AMD GPUs, and an FPGA platform.

## CNN and MNIST

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~87%` on the provided test dataset.

The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## Local Development Environment

The project requires a CUDA-supported operating system, C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler.

### How to Build

There are two options to build this project, the first is using the [Hunter] package manager and the other is using [Docker](https://www.docker.com/). We sugguest using CMake along with Hunter, but it is known not to work on all operating systems. In this case, we suggest that you either using Docker or install the libraries needed (mainly `HDF5`).

#### Using Hunter Package Manager

By default, the compilation uses the [Hunter] --- a C package manager. This method requires that you have the CUDA toolkit installed on your machine.

Assuming that you have checked out the project into `$SRCDIR` do

```{.sh}
cd $SRCDIR
mkdir build
cd build
cmake $SRCDIR
```

This will download the required software needed for the project (see the [hunter docs][hunterdoc] for more information). You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to build the project.

```{.sh}
make
```

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the [`CMakeLists.txt`] and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section. Documentation on the CMake commands is found in the [documentation page][cmakedoc].

#### Using Docker Container

[![Docker Automated build](https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg)](https://hub.docker.com/r/webgpu/ece408project/)

Also included is a [Docker](http://docker.io/) build file. This file is a specification for a Docker container image. It can be used to build and launch a container (think of a virtual machine) which contains this project along with all the software required to run it. Using a GPU within Docker is only supported on Linux(you can compile and run the serial code on any operating system), and we recommend using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image. To build the Docker container, do

```{.sh}
cd $SRCDIR
docker build . -t ece408project
```

Once built, the `ece408project` image would be listed by the `docker images` command. This will compile your project. You can launch the docker image using

```{.sh}
docker run -it ece408project
```

## How to Run Code

Test your implementation with small batch size first to verify the correctness. You can parse the `data/test100.hdf5` into smaller chunks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in `data/test2.hdf5`, `data/test10.hdf5` and `data/test100.hdf5` in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
```


## Reporting Issues

Please use the [GitHub issue manager] to report any issues or suggestions about the project.

## Resources Used
- [University of Illinois: ECE 408 staff][ece408]
- [Fei Deng][Fei Deng]
- [Cmake Documentation][cmakedoc]
- [Hunter][hunter]
- [Hunter Documentation][hunterdoc]
- [Rancehpp][rangehpp]


[github issue manager]: https://github.com/Atrifex/Code-Acceleration/issues

[ece408]: https://github.com/webgpu/ece408project/

[Fei Deng]: https://gitlab.engr.illinois.edu/feideng2/ece408_project_public

[cmakedoc]: https://cmake.org/cmake/help/latest/

[hunterdoc]: https://docs.hunter.sh/en/latest/

[rangehpp]: https://github.com/harrism/cpp11-range

[hunter]: https://github.com/ruslo/hunter

