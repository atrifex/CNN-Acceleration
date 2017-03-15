# ECE 408 Project

The goal of this project is to accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using GPUs. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://wiki.illinois.edu/wiki/display/ece408f16/Book+Chapters?preview=/602518692/603851747/3rd-Edition-Chapter16-case-study-DNN-FINAL.pdf). The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## CNN and MNIST

Read the book chapter and familiarize youself with the CNN algorithm.

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~87%` on the provided test dataset.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format and we have provided the code to read the input model and the training dataset.

## CUDA Implementation

Book chapters 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of convolutional layer and possible optimization. Your CUDA implementation would be evaluated based on performance and accuracy. Apply any optimization you think would bring benefit and feel free to modify any part of the code. You should not use `cuBLAS` or `cuDNN` for the implementation, but you are expected to compare your implementation with those libraries --- profiling the code as well as comparing the algorithms used (if algorithm information is publically available).

## Remote Development Environment

The easiest way to develop the project is to use rai through the following prebuilt binaries. The stable version only supports Linux and OSX. For students with Windows, you can try the beta version or use the Linux on [EWS](http://it.engineering.illinois.edu/ews) for RAI.

**NOTE:** Even if you use your local development environment, your final code must run within the RAI system. Also, your final report performance measurements must be done within RAI.

### Download Binaries

The code is continuously built and published. The client can be downloaded from the following URLs (depending on your OS and Architecture):

| Operating System | Architecture | Stable Version Link                                                    | Development Version Link                                                    |
| ---------------- | ------------ | ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Linux            | i386         | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-386.tar.gz)    | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-386.tar.gz)     |
| Linux            | amd64        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-amd64.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-amd64.tar.gz)   |
| Linux            | armv5        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-armv5.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-armv5.tar.gz)   |
| Linux            | armv6        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-armv6.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-armv6.tar.gz)   |
| Linux            | armv7        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-armv7.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-armv7.tar.gz)   |
| Linux            | arm64        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-linux-arm64.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-linux-arm64.tar.gz)   |
| OSX/Darwin       | i386         | [URL](http://rai-server.s3.amazonaws.com/dist/rai-darwin-386.tar.gz)   | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-darwin-386.tar.gz)    |
| OSX/Darwin       | amd64        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-darwin-amd64.tar.gz) | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-darwin-amd64.tar.gz)  |
| Windows          | i386         | [URL](http://rai-server.s3.amazonaws.com/dist/rai-windows-386.tar.gz)  | [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-windows-386.tar.gz)   |
| Windows          | amd64        | [URL](http://rai-server.s3.amazonaws.com/dist/rai-windows-amd64.tar.gz)| [URL](http://rai-server.s3.amazonaws.com/dist/dev/rai-windows-amd64.tar.gz) |

### Client

#### Set up your Secret Key

Each team will be contacted by a TA and given a secret key to use this service. Do not share your key with other teams. The secret key is used to authenticate you with the server.

The `RAI_SECRET_KEY`, `RAI_TEAM_NAME`, and `RAI_ACCESS_KEY` should be specified in your `~/.rai.profile` (linux/OSX) or `%HOME%/.rai.profile` (Windows -- for me this is `C:\Users\abduld\.rai.profile`) in the following way.

```bash
RAI_TEAM_NAME='Your Team Name Here'
RAI_USER_NAME='user'
RAI_ACCESS_KEY='XXXXXXXX'
RAI_SECRET_KEY='XXXXX'
```

The above will need to match the email you recieved from `postmaster@webgpu.com` on Nov 23. If you did not recieve the email, then contact the TA. Also, contact the TA with your team name as soon as possible.  Do not share your keys with other users or teams. The access and secret key is used to authenticate you with the server. Both the team name and the username are used to identify you to the system.

#### Run the Client

To run the client, use

```bash
rai -d <project folder>
```

From a user's point a view when the client runs, the local directory specified by `-d` gets uploaded to the server and extracted into the `/src` directory on the server. The server then executes the build commands from the `rai-build.yml` specification within the `/build` directory. Once the commands have been run, or there is an error, a zipped version of that `/build` directory is available from the server for download.

The server limits the task time to be an hour with a maximum of 8GB of memory being used within a session. The output `/build` directory is only available to be downloaded from the server for a short amount of time. Networking is also disabled on the execution server.

#### Internal Details (Ignore if not Interested)

The client sends job submission requests to the rai server. The internal steps the client takes are as follows:

1.  The client creates an archive of your directory and posts it to Amazon S3
2.  The client creates a unique identifier (here called `ID`). These IDs are generated using [`NewObjectId`](https://godoc.org/labix.org/v2/mgo/bson#NewObjectId).
3.  The client creates a job request and publishes to the `tasks` topic on the queue. The job request has the ID field with the value `ID` and is mashaled using using the [`bson`](https://godoc.org/labix.org/v2/mgo/bson) library. The reason for using `bson` is that we will want to store the results in mongodb in the future.
4.  The client subscribes to the topic `log-ID` and prints the results on that topic.
5.  The client stops listening when the message on the topic has a tag `TagEnd`.

### Project Build Sepecification

The `rai-build.yml` must exist in your project directory. If not available, then the system will use the default build script. In some cases you may not be able to execute certain commands, in this senario the current workaround is to create a bash file and insert the commands you need to run. You can then execute the bash script within `rai-build.yml`.

The `rai-build.yml` is written as a [Yaml](http://yaml.org/) ([Spec](http://www.yaml.org/spec/1.2/spec.html)) file and has the following structure.

```yaml
rai:
  version: 0.1 # this is required
  image: webgpu/rai:root # this is ignored at this moment with the webgpu/rai:root
                         # image being used by default. webgpu/rai:root is a docker
                         # image which can be viewed at https://hub.docker.com/r/webgpu/rai/
resources:
  gpus: 1 # currently this field is ignored, but in the future you'd be able to specify your
          # system requirements
commands:
  build:
    - echo "Building project"
    # Since the system already contains the dependencies (like HDF5 and ZLib) we do not
    # need the hunter package manager. This speeds up the compilation as well
    - cmake -DCONFIG_USE_HUNTER=OFF /src
    # Run the make file to compile the project.
    - make
    # here we break the long command into multiple lines. The Yaml
    # format supports this using a block-strip command. See
    # http://stackoverflow.com/a/21699210/3543720 for info
    - >-
      nvprof --analysis-metrics --print-api-trace --cpu-profiling on
      --demangling on --export-profile profile.nvvp
      --force-overwrite --log-file run.log --print-gpu-trace
      -- ./ece408 /src/data/test10.hdf5 /src/data/model.hdf5 10
```

Syntax errors will be reported and the job will not be executed. You can check if your file is in a valid yaml format by using tools such as [Yaml Validator](http://codebeautify.org/yaml-validator).

## Profiling

Profiling can be performed using `nvprof`. Place the following build commands in your `rai-build.yml` file

```yaml
    - >-
      nvprof --cpu-profiling on --export-profile timeline.nvprof --
      ./ece408 /src/data/test10.hdf5 /src/data/model.hdf5 10
    - >-
      nvprof --cpu-profiling on --export-profile analysis.nvprof --analysis-metrics --
      ./ece408 /src/data/test10.hdf5 /src/data/model.hdf5 10
```

You could change the input and test datasets. This will output two files `timeline.nvprof` and `analysis.nvprof` which can be viewed using the `nvvp` tool (by performing a `file>import`).

_NOTE:_ `nvvp` will only show performance metrics for GPU invocations, so it may not show any analysis when you only have serial code.

### Project Submission

You will use the same client (with certain options) for the final submission. The submission system notify the teaching assistants and record your ranking. You will need the above credentials to make your final submission.

To submit your project, run

```bash
rai submit -d <project folder>
```

To perform the final project submission, you must have the `USAGE`, `README`, and `report.pdf` files in your project folder (as stated in the ["What to Deliver"](#what-to-deliver) section). The submission system ignores your `rai-build.yml` file and instead runs the following build file:

```yaml
rai:
  version: 0.1
resources:
  gpus: 1
commands:
  build:
    - echo "Submitting project"
    - cp -r /src /build/submission_code
    - cmake -DCONFIG_USE_HUNTER=OFF /src
    - make
    - /usr/bin/time ./ece408 /src/data/testfull.hdf5 /src/data/model.hdf5 10000
```

**NOTE::** Only your last submission is recorded, so please make sure that your last submission is the one you'd want to be graded.

### Competition Rankings

You can see the current rankings for the project competition by invoking

```bash
rai rankings
```

You can see only the top 10 teams by invoking


```bash
rai rankings -l 10
```


### Reporting Issues


If emailing the TA with a problem, then please include the output of

```
rai version
```

as well as the output of

```
rai buildtime
```

you can also invoke the rai command with verbose and debug outputs using

```
rai --verbose --debug
```

## Local Development Environment

**NOTE:** Even if you use your local development environment, your final code must run within the RAI system. Also, your final report performance measurements must be done within RAI.

The project requires a CUDA-supported operating system, C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler. On windows, we require Visual Studio 2015 (Service Pack 3) which you can download from the webstore. For other systems, a CUDA compatible compiler is required (e.g. for OSX the [clang compiler](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#system-requirements) is the only one supported).

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

This will download the required software needed for the project (see the [hunter docs][hunterdoc] for more information). You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to buidl the project.

```{.sh}
make
```

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the [`CMakeLists.txt`](https://github.com/webgpu/ece408project/blob/master/CMakeLists.txt) and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section. Documentation on the CMake commands is found in the [documentation page][cmakedoc].

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

### Running the Serial Code

```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 batch_size
```

the `batch_size` must match the size of the dataset. If `batch_size` is unspecified, the default value is dependent on the input (10 for "../data/test10.hdf5", ..., 10000 for "../data/testfull.hdf5"), which is also the size of `data.hdf5`.

## How to Test

Test your implementation with small batch size frist to verify the correctness. You can parse the `data/test100.hdf5` into smaller chunks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in `data/test2.hdf5`, `data/test10.hdf5` and `data/test100.hdf5` in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
```

## What to Deliver

A `.tar.gz` file which contains the report, code directory, the build scripts, and, possibly, the input dataset needs to be delivered to the Teaching Assistants.

-   Code:  A `USAGE` file needs to be placed included in the archive file which includes instructions on how to compile and run your code. If the report performs any profiling, the `USAGE` file must also specify how to run the performance measurements.
-   Report: A PDF version report must be included within the `.tar.gz` file. The report should describe and evaluate the optimizations you tried. The report does not have a page limit, but as usual, you should strive to be thorough, concise, and quantitative in your performance analysis.
    The report must be named `report.pdf`

Make sure you have a working CUDA implementation before applying any optimizations.

## Optimization Opportunities

The serial version of the code is amicable to many optimization opportunities, the following is an incomplete set of them:

-   Optimize the CUDA memory copies to decrease the overhead of memory transfers
-   Overlapping the memory transfer and the compute and/or independent computations using CUDA streams
-   Performing layout transformations to get coallessed accesses or to make better use of the cache
-   Using low precision to perform the computation (for example using `float16` or binary values)
-   Based on the size of the convolution, utilitize better algorithms to perform the computation (for example using the [Winograd Kernel][https://www.nervanasys.com/winograd-2/])

## Utility Functions

We provide a some helper utility functions in the [`utils.hpp`][utilshpp] file.

### How to Time

In [`utils.hpp`][utilshpp] a function called `now()` which allows you to get the current time at a high resolution. To measure the overhead of a function `f(args...)`, the pattern to use is:

```{.cpp}
const auto tic = now();
f(args...);
const auto toc = now();
const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();;
std::cout << "Calling f(args...) took " << elapsed << "milliseconds\n";
```

### Range For Loops

Throughout the serial code, we use the [`range.hpp`][rangehpp] to make the code easier to understand. Essentially,

```{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
```

Is equivalent to

```{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
```

### Checking Errors

To check for CUDA errors, specialize the `check_success` function in `utils.hpp` to also handle `cudaError_t`. For example:

```{.cpp}
template <>
bool check_success<cudaError_t>(const cudaError_t &err) {
  const auto res = err == cudaSuccess;
  if (res == true) {
    return res;
  }
  std::cout << "Failed in CUDA. Error = " << cudaGetErrorString(err) << std::endl;
  assert(res);
  return res;
}
```

`check_success` can then be used when calling CUDA functions:

```{.cpp}
check_success(cudaFree(deviceData));
```

## Reporting Issues

Please use the [Github issue manager] to report any issues or suggestions about the project.

[utilshpp]: https://github.com/webgpu/ece408project/blob/master/src/utils.hpp

[cmakedoc]: https://cmake.org/cmake/help/latest/

[hunterdoc]: https://docs.hunter.sh/en/latest/

[rangehpp]: https://github.com/harrism/cpp11-range

[github issue manager]: https://github.com/webgpu/ece408project/issues

[hunter]: https://github.com/ruslo/hunter
