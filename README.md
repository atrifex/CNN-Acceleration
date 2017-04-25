# Code Acceleration

The goal of this project is two fold. First, accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using CUDA and OpenCL. Second, evaluate its performance on NVIDIA, AMD GPUs, and an FPGA platform.

## CNN and MNIST

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~87%` on the provided test dataset.

The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## System Requirements And Building the Project

The project requires a C++ compiler, CUDA 8 Toolkit, and OpenCL 1.2 or higher. 

The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).
Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler.

### Building CUDA and OpenCL versions

To build the CUDA and OpenCL versions the [Hunter] package manager with Cmake needs to be used. Install the libraries needed (mainly `HDF5`).

Assuming that you have checked out the project into `$PROJECT_DIR`, execute the following sequence of commands:

```{.sh}
cd $PROJECT_DIR/cuda_implementation/
mkdir build
cd build
cmake ../
```

This will download the required software needed for the project (see the [hunter docs][hunterdoc] for more information). You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to build the project.

```{.sh}
make
```

The same sequence of commands need to be executed in the openCL_implementation directory to run the OpenCL version.

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the [`CMakeLists.txt`] and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section. Documentation on the CMake commands is found in the [documentation page][cmakedoc].

### Building FPGA Version with OpenCL Standard
Since the FPGA version needs to use modified versions of the OpenCL libraries provided by Intel, it makes more sense to just use the Makefile provided by Intel.

The first step in building is figuring out which boards are available. The following command will list all of the available boards:
```{.sh}
aoc --list-boards
```

#### Compiling the OpenCL Kernel to Run on Board
```{.sh}
cd $PROJECT_DIR/fpgaOpenCL_implementation/
aoc device/cnn.cl -o bin/cnn.aocx --board <board>
```

#### Compiling the OpenCL Kernel for Emulator
```{.sh}
cd $PROJECT_DIR/fpgaOpenCL_implementation/
aoc -march=emulator device/cnn.cl -o bin/cnn.aocx --board <board>
```

#### Compiling the OpenCL Host Code
Assuming that you have checked out the project into `$PROJECT_DIR`, execute the following sequence of commands:
```{.sh}
cd $PROJECT_DIR/fpgaOpenCL_implementation/
make
```

This will generate a bin folder that will contain the executable that can run the FPGA version of the code.

## How to Run Code

Test your implementation with small batch size first to verify the correctness. You can parse the `data/test100.hdf5` into smaller chunks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in `data/test2.hdf5`, `data/test10.hdf5` and `data/test100.hdf5` in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

To run the code, you need to be in the build or bin directories of a given version of the project.

### CUDA Version
```{.sh}
./cuda_CNN ../../data/test10.hdf5 ../../data/model.hdf5 10
```
### OpenCL Version
```{.sh}
./openCL_CNN ../../data/test10.hdf5 ../../data/model.hdf5 10
```

### FPGA Version on Board
```{.sh}
./fpga_CNN ../../../data/test10.hdf5 ../../../data/model.hdf5 10
```

### FPGA Version with Emulator
```{.sh}
CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./fpga_CNN ../../../data/test10.hdf5 ../../../data/model.hdf5 10
```

## Results
The following are the results on a Tesla P100 GPU as of 4/22/2017.

### CUDA Version
test10: 15.9631 milliseconds

test100: 17.6193 milliseconds

testfull: 524.852 milliseconds

### OpenCL Version
test10: 3.28717 milliseconds

test100: 3.76949 milliseconds

testfull: 34.1698 milliseconds

### FPGA Version on Board
Need to get access to system with an FPGA board to test this version of the code.

### FPGA Version with Emulator
test10: 7423.32 milliseconds

test100: 68188.2 milliseconds

testfull: TIMES OUT

Note: Major optimizations need to be made before the results from the FPGA version can match the results from the OpenCL and CUDA versions of the code.


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

