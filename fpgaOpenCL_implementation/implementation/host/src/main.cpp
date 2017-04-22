
#include "cnn.h"
using namespace aocl_utils;

static unsigned int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// converlution and fully forward dimensions
static unsigned int conv1_dims[4] = { CONV1_OUTPUT_CHANNELS, CONV1_INPUT_CHANNELS, CONV_ROWS, CONV_COLS };
static unsigned int conv2_dims[4] = { CONV2_OUTPUT_CHANNELS, CONV2_INPUT_CHANNELS, CONV_ROWS, CONV_COLS };
static unsigned int fc1_dims[2] = { FC1_ROWS, FC1_COLS };
static unsigned int fc2_dims[2] = { FC2_ROWS, FC2_COLS };

// reference dimensions
static unsigned int ref_dims[2] = { FLAGS_batch_size, NUM_DIGITS };

// input dimensions
static unsigned int input_dims[4] = { FLAGS_batch_size, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS };
static unsigned int input_unroll_dims[4] = { FLAGS_batch_size, INPUT_CHANNELS, CONV_ROWS * CONV_COLS, A_ROWS * A_COLS };

// forward operation intermediate dimensions
static unsigned int a_dims[4] = { FLAGS_batch_size, CONV1_OUTPUT_CHANNELS, A_ROWS, A_COLS };
static unsigned int b_dims[4] = { FLAGS_batch_size, CONV1_OUTPUT_CHANNELS, B_ROWS, B_COLS };
static unsigned int b_unroll_dims[4] = { FLAGS_batch_size, CONV1_OUTPUT_CHANNELS, CONV_ROWS * CONV_COLS, C_ROWS * C_COLS };
static unsigned int c_dims[4] = { FLAGS_batch_size, CONV2_OUTPUT_CHANNELS, C_ROWS, C_COLS };
static unsigned int d_dims[4] = { FLAGS_batch_size, CONV2_OUTPUT_CHANNELS, D_ROWS, D_COLS };
static unsigned int d_dims2[2] = { FLAGS_batch_size, FC1_ROWS };
static unsigned int e_dims[2] = { FLAGS_batch_size, FC2_ROWS };
static unsigned int f_dims[2] = { FLAGS_batch_size, NUM_DIGITS };

// flatterened length of all the dimensions above
static unsigned int conv1_len;
static unsigned int conv2_len;
static unsigned int fc1_len;
static unsigned int fc2_len;

static unsigned int input_len;
static unsigned int input_unroll_len;

static unsigned int a_len;
static unsigned int b_len;
static unsigned int b_unroll_len;
static unsigned int c_len;
static unsigned int d_len;
static unsigned int e_len;
static unsigned int f_len;

static unsigned int output_len;

// pointers to the device memory used in the forward operation
cl_mem conv1_device_;
cl_mem conv2_device_;
cl_mem fc1_device_;

cl_mem conv1_device;
cl_mem conv2_device;
cl_mem fc1_device;
cl_mem fc2_device;

cl_mem input_device;
cl_mem input_unroll_device;

cl_mem a_device;
cl_mem b_device;
cl_mem b_unroll_device;
cl_mem c_device;
cl_mem d_device;
cl_mem e_device;
cl_mem f_device;

cl_mem output_device;

cl_platform_id platform;                        // OpenCL platform
cl_device_id device;                            // device ID
cl_context context;                             // context
cl_command_queue queues[NUM_CMD_QUEUES];        // command queue
cl_program program;                             // program
map<string, cl_kernel> kernels;                 // kernels

const char *getErrorString(cl_int error){
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void cleanUp(){
    for(int i = 0; i < NUM_CMD_QUEUES; i++){
        if (queues[i] != 0)
            clReleaseCommandQueue(queues[i]);
    }

    for (const auto & kernel : kernels) {
        if (kernel.second != 0)
            clReleaseKernel(kernel.second);
    }

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

    clReleaseMemObject(conv1_device_);
    clReleaseMemObject(conv2_device_);
    clReleaseMemObject(fc1_device_);

    clReleaseMemObject(conv1_device);
    clReleaseMemObject(conv2_device);
    clReleaseMemObject(fc1_device);
    clReleaseMemObject(fc2_device);

    clReleaseMemObject(input_device);
    clReleaseMemObject(input_unroll_device);

    clReleaseMemObject(a_device);
    clReleaseMemObject(b_device);
    clReleaseMemObject(b_unroll_device);
    clReleaseMemObject(c_device);
    clReleaseMemObject(d_device);
    clReleaseMemObject(e_device);
    clReleaseMemObject(f_device);

    clReleaseMemObject(output_device);
}

inline void checkErr(cl_int error, const int line){
    if (error != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  getErrorString(error)  << std::endl;
        std::cerr << "At line: " <<  line  << std::endl;
        cleanUp();
        exit(EXIT_FAILURE);
    }
}

cl_program createProgram(cl_context context, cl_device_id device, const char* fileName){
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()){
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &errNum);
    if (program == NULL) {
        checkErr(errNum, __LINE__);
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

void createKernel(string kernel_name){
    // error code returned from api calls
    cl_int err;

    // create kernel and check for errors
    kernels[kernel_name] = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, __LINE__);
}

void initializeOpenCLParameters(){
    cl_int err;

    if(!setCwdToExeDir()) {
      return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA");
    if(platform == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }
    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &err);
    checkErr(err, __LINE__);

    // Create all of the command queues
    for (unsigned int i = 0; i < NUM_CMD_QUEUES; i++) {
        queues[i] = clCreateCommandQueue(context, device, 0, &err);
        checkErr(err, __LINE__);
    }

    // Load source code for program
    if ((program = createProgram(context, device, "cnn.cl")) == NULL) {
        cleanUp();
        std::cerr << "Program creation failed." << std::endl;
        exit(1);
    }

{
    createKernel("transform_conv1");
    createKernel("transform_conv2");
    createKernel("transform_fc1");
    createKernel("unroll1");
    createKernel("unroll2");
    createKernel("matrix_multiplication1");
    createKernel("matrix_multiplication2");
    createKernel("average_pool1");
    createKernel("average_pool2");
    createKernel("fully_forward1");
    createKernel("fully_forward2");
    createKernel("arg_max");
}

}

bool init() {
  // Create the program.
  std::string binary_file = getBoardBinaryFile("hello_world", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel_name = "hello_world";  // Kernel name, as defined in the CL file
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  return true;
}

/*
    load_data(float *x, float *y)
    DESCRIPTION:
        Read the data from test file.
    INPUT:
        x - pointer to the input array
        y - pointer to the reference array
    RETURN:
        0 - success
        1 - fail
*/
static unsigned int load_data(float *x, float *y) {

    // Open the data file
    const auto file_id = H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Open the dataset x and y
    const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
    const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

    // Get the dataset x dimensions
    const auto x_space = H5Dget_space(x_id);
    const auto x_ndims = H5Sget_simple_extent_ndims(x_space);
    assert(x_ndims == 4);

    hsize_t *x_dims = allocate<hsize_t>(x_ndims);
    H5Sget_simple_extent_dims(x_space, x_dims, NULL);
    if (x_dims[0] != FLAGS_batch_size) {
        std::cout << "Data size does not match batch size specified!\n";

        // return error
        delete[] x_dims;
        return 1;
    }
    std::cout << "Input dimensions = " << x_dims[0] << " x " << x_dims[1] << " x " << x_dims[2] << " x " << x_dims[3] << "\n";

    // Read the dataset x and y
    check_success(H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
    check_success(H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

    // Close the dataset x and y
    check_success(H5Dclose(x_id));
    check_success(H5Dclose(y_id));

    // Close the file
    check_success(H5Fclose(file_id));

    // return success
    delete[] x_dims;
    return 0;
}

/*
    load_model(float *conv1, float *conv2, float * fc1, float *fc2)
    DESCRIPTION:
        Read the data from model file.
    INPUT:
        conv1 - pointer to the conv1 array
        conv2 - pointer to the conv2 array
        fc1 - pointer to the fc1 array
        fc2 - pointer to the fc2 array
*/
static void load_model(float *conv1, float *conv2, float * fc1, float *fc2) {

    // Open the model file
    const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Open the dataset
    const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
    const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
    const auto fc1_id = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
    const auto fc2_id = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

    // Read the dataset
    check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv1));
    check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv2));
    check_success(H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
    check_success(H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

    // Close the dataset x and y
    check_success(H5Dclose(conv1_id));
    check_success(H5Dclose(conv2_id));
    check_success(H5Dclose(fc1_id));
    check_success(H5Dclose(fc2_id));

    // Close the file
    check_success(H5Fclose(file_id));
}

/*
    get_ground_truth(const float *input, unsigned int *output)
    DESCRIPTION:
        Get the reference result (ground truth).
    INPUT:
        input - pointer to the ref array
        output - pointer to the ground_truth array
*/
static void get_ground_truth(const float *input, unsigned int *output) {
    for (unsigned int i = 0; i < ref_dims[0]; i++) {
        unsigned int max_idx = 0;
        float max = input[i * ref_dims[1]];
        for (unsigned int j = 0; j < ref_dims[1]; j++) {
            const float elem = input[(i * ref_dims[1]) + j];
            if (elem > max) {
                max_idx = j;
                max = elem;
            }
        }
        output[i] = max_idx;
    }
}

/*
    forward_operation(const float *input, const float *conv1, const float *conv2, const float * fc1, const float *fc2, unsigned int *output)
    DESCRIPTION:
        Forward operation for the CNN, a combination of conv layer + average pooling + relu.
    INPUT:
        input - pointer to the input array
        conv1 - pointer to the conv1 array
        conv2 - pointer to the conv2 array
        fc1 - pointer to the fc1 array
        fc2 - pointer to the fc2 array
        output - pointer to the output array
*/
void forward_operation(const float *input, const float *conv1, const float *conv2, const float * fc1, const float *fc2, unsigned int *output) {

    // since the provided dataset has some very weird dimensions, we transform them to more human understandable ones
    size_t lws_conv[] = {CONV_COLS, CONV_ROWS, CONV1_OUTPUT_CHANNELS};

    /***** transform_conv1 *****/
    size_t gws_conv1[] = {CONV1_INPUT_CHANNELS*lws_conv[0],1*lws_conv[1],1*lws_conv[2]};

    // copying over conv1 buffer
    checkErr(clEnqueueWriteBuffer(queues[QUEUE_IDX_CONV1], conv1_device_, CL_FALSE, 0, conv1_len * sizeof(float),
            (void*)conv1, 0, NULL, NULL), __LINE__);

    // setting arguments and calling transform_conv1 kernel
    checkErr(clSetKernelArg(kernels["transform_conv1"], 0, sizeof(cl_mem), &conv1_device_), __LINE__);
    checkErr(clSetKernelArg(kernels["transform_conv1"], 1, sizeof(cl_mem), &conv1_device), __LINE__);
    checkErr(clEnqueueNDRangeKernel(queues[QUEUE_IDX_CONV1], kernels["transform_conv1"], 3, NULL, gws_conv1, lws_conv, 0, NULL, NULL), __LINE__);

    /***** transform_conv2 *****/
    size_t gws_conv2[] = {CONV2_OUTPUT_CHANNELS*lws_conv[0],1*lws_conv[1],1*lws_conv[2]};

    // copying over conv2 buffer
    checkErr(clEnqueueWriteBuffer(queues[QUEUE_IDX_CONV2], conv2_device_, CL_FALSE, 0, conv2_len * sizeof(float),
            (void*)conv2, 0, NULL, NULL), __LINE__);

    // setting arguments and calling transform_conv2 kernel
    checkErr(clSetKernelArg(kernels["transform_conv2"], 0, sizeof(cl_mem), &conv2_device_), __LINE__);
    checkErr(clSetKernelArg(kernels["transform_conv2"], 1, sizeof(cl_mem), &conv2_device), __LINE__);
    checkErr(clEnqueueNDRangeKernel(queues[QUEUE_IDX_CONV2], kernels["transform_conv2"], 3, NULL, gws_conv2, lws_conv, 0, NULL, NULL), __LINE__);

    /***** transform_fc1 *****/
    size_t lws_fc1[] = {d_dims[3], d_dims[2], d_dims[1]};
    size_t gws_fc1[] = {FC1_COLS*lws_fc1[0],lws_fc1[1],lws_fc1[2]};

    // copying over fc1 buffer
    checkErr(clEnqueueWriteBuffer(queues[QUEUE_IDX_FC1], fc1_device_, CL_FALSE, 0, fc1_len * sizeof(float),
            (void*)fc1, 0, NULL, NULL), __LINE__);

    // setting arguments and calling transform_conv2 kernel
    checkErr(clSetKernelArg(kernels["transform_fc1"], 0, sizeof(cl_mem), &fc1_device_), __LINE__);
    checkErr(clSetKernelArg(kernels["transform_fc1"], 1, sizeof(cl_mem), &fc1_device), __LINE__);
    checkErr(clEnqueueNDRangeKernel(queues[QUEUE_IDX_FC1], kernels["transform_fc1"], 3, NULL, gws_fc1, lws_fc1, 0, NULL, NULL), __LINE__);

    // Copy fc2 into device memory
    checkErr(clEnqueueWriteBuffer(queues[QUEUE_IDX_FC2], fc2_device, CL_FALSE, 0, fc2_len * sizeof(float),
            (void*)fc2, 0, NULL, NULL), __LINE__);

    // q is the index for queues, reset to 0 for every kernel lunch such that all kernels will be streamlined
    // i.e. a new kernel that is using queue x will only use the data calculated by the previous kernel that is also using queue x
    unsigned int q = 0;

    // batch num is how much data we will be using for each kernel, need to change only once for unroll2
    unsigned int batch_num = BATCH_NUM_PER_STREAM;

    // unroll kernel 1, this will unroll from input dataset, also use async memcpy to make it more efficient
    for (unsigned int start = 0; start < input_dims[0]; start += batch_num) {
        const unsigned int wgGlobSize = min(batch_num, input_dims[0] - start);
        const unsigned int offset = start * INPUT_NUM_ELEMENTS;

        // copy over batch of memory to be worked on
        checkErr(clEnqueueWriteBuffer(queues[q], input_device, CL_FALSE, offset, wgGlobSize * INPUT_NUM_ELEMENTS * sizeof(float),
                (void*)(input + offset), 0, NULL, NULL), __LINE__);

        // setting arguments and calling unroll1 kernel
        size_t lws_unroll1[] = {BLOCK_SIZE};
        size_t gws_unroll1[] = {wgGlobSize*lws_unroll1[0]};
        checkErr(clSetKernelArg(kernels["unroll1"], 0, sizeof(cl_mem), &input_device), __LINE__);
        checkErr(clSetKernelArg(kernels["unroll1"], 1, sizeof(cl_mem), &input_unroll_device), __LINE__);
        checkErr(clSetKernelArg(kernels["unroll1"], 2, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["unroll1"], 1, NULL, gws_unroll1, lws_unroll1, 0, NULL, NULL), __LINE__);

        // increment queue index
        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // matrix multiplication kernel 1
    q = 0;
    size_t lws_mm[] = {HALF_TILE_SIZE, HALF_TILE_SIZE, 1};
    clFinish(queues[QUEUE_IDX_CONV1]);
    for (unsigned int start = 0; start < input_unroll_dims[0]; start += batch_num) {
        size_t gws_mm1[] = {MATRIX_MUL1_BLOCKS_PER_NUM*lws_mm[0], min(batch_num, input_unroll_dims[0] - start)*lws_mm[1], 1*lws_mm[2]};

        // setting arguments and calling matrix_multiplication1 kernel
        checkErr(clSetKernelArg(kernels["matrix_multiplication1"], 0, sizeof(cl_mem), &conv1_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication1"], 1, sizeof(cl_mem), &input_unroll_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication1"], 2, sizeof(cl_mem), &a_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication1"], 3, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["matrix_multiplication1"], 2, NULL, gws_mm1, lws_mm, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // average pool kernel 1, err, like the name suggests, it averages stuff...
    q = 0;
    unsigned int layers = MAX_THREADS_PER_BLOCK / B_NUM_ELEMENTS;
    size_t lws_avg1[] = {b_dims[2], b_dims[3], layers};
    for (unsigned int start = 0; start < a_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, a_dims[0] - start);
        const unsigned int wgGlobSize = static_cast<unsigned int>(ceil(todo_count / (float)layers));
        const unsigned int offsetFromStart = start + todo_count;

        size_t gws_avg1[] = {wgGlobSize*lws_avg1[0], 1*lws_avg1[1], 1*lws_avg1[2]};

        // setting arguments and calling average_pool1 kernel
        checkErr(clSetKernelArg(kernels["average_pool1"], 0, sizeof(cl_mem), &a_device), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool1"], 1, sizeof(cl_mem), &b_device), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool1"], 2, sizeof(cl_uint), &start), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool1"], 3, sizeof(cl_uint), &offsetFromStart), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["average_pool1"], 3, NULL, gws_avg1, lws_avg1, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // unroll kernel 2, now we have 32 channels per number, so multiply the batch number by 32
    // and we don't need to copy data from anywhere
    q = 0;
    batch_num = batch_num * BATCH_NUM_FACTOR;
    layers = UNROLL2_LAYERS;
    const unsigned int total_channels = b_dims[0] * b_dims[1];
    size_t lws_unroll2[] = {b_unroll_dims[3], layers, 1};
    for (unsigned int start = 0; start < total_channels; start += batch_num) {
        size_t gws_unroll2[] = {(min(static_cast<unsigned int>(ceil(batch_num / (float)layers)),
                                static_cast<unsigned int>(ceil(total_channels - start) / (float)layers))*lws_unroll2[0]),
                                1*lws_unroll2[1], 1*lws_unroll2[2]};

        // setting arguments and calling unroll2 kernel
        checkErr(clSetKernelArg(kernels["unroll2"], 0, sizeof(cl_mem), &b_device), __LINE__);
        checkErr(clSetKernelArg(kernels["unroll2"], 1, sizeof(cl_mem), &b_unroll_device), __LINE__);
        checkErr(clSetKernelArg(kernels["unroll2"], 2, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["unroll2"], 3, NULL, gws_unroll2, lws_unroll2, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // matrix multiplication kernel 2, again multiply two matrices
    q = 0;
    batch_num = batch_num / BATCH_NUM_FACTOR;
    clFinish(queues[QUEUE_IDX_CONV2]);
    for (unsigned int start = 0; start < b_unroll_dims[0]; start += batch_num) {
        size_t gws_mm2[] = {(min(static_cast<unsigned int>(ceil(batch_num / 2.0f)),
                            static_cast<unsigned int>(ceil((b_unroll_dims[0] - start) / 2.0f)))*lws_mm[0]),
                            1*lws_mm[1], 1*lws_mm[2]};

        // setting arguments and calling matrix_multiplication2 kernel
        checkErr(clSetKernelArg(kernels["matrix_multiplication2"], 0, sizeof(cl_mem), &conv2_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication2"], 1, sizeof(cl_mem), &b_unroll_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication2"], 2, sizeof(cl_mem), &c_device), __LINE__);
        checkErr(clSetKernelArg(kernels["matrix_multiplication2"], 3, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["matrix_multiplication2"], 2, NULL, gws_mm2, lws_mm, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // average pool kernel 2, really, you need to read this to understand average?
    q = 0;
    layers = BLOCK_SIZE / D_NUM_ELEMENTS;
    size_t lws_avg2[] = {d_dims[2], d_dims[3], layers};
    for (unsigned int start = 0; start < c_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, c_dims[0] - start);
        const unsigned int wgGlobSize = static_cast<unsigned int>(ceil(todo_count / (float)layers));
        const unsigned int offsetFromStart = start + todo_count;

        size_t gws_avg2[] = {wgGlobSize*lws_avg2[0], 1*lws_avg2[1], 1*lws_avg2[2]};

        // setting arguments and calling average_pool2 kernel
        checkErr(clSetKernelArg(kernels["average_pool2"], 0, sizeof(cl_mem), &c_device), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool2"], 1, sizeof(cl_mem), &d_device), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool2"], 2, sizeof(cl_uint), &start), __LINE__);
        checkErr(clSetKernelArg(kernels["average_pool2"], 3, sizeof(cl_uint), &offsetFromStart), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["average_pool2"], 3, NULL, gws_avg2, lws_avg2, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // fully forward kernel 1, even though it's named different, but deep down it's still just matrix multiplication...
    q = 0;
    clFinish(queues[QUEUE_IDX_FC1]);
    size_t lws_ff1[] = {lws_mm[0], lws_mm[1], lws_mm[2]};
    for (unsigned int start = 0; start < d_dims2[0]; start += batch_num) {
        const unsigned int a_height = min(batch_num, d_dims2[0] - start);

        size_t gws_ff1[] = {((static_cast<unsigned int>(ceil(a_height / (float)TILE_SIZE)))*lws_ff1[0]), 1*lws_ff1[1], 1*lws_ff1[2]};

        // setting arguments and calling fully_forward1 kernel
        checkErr(clSetKernelArg(kernels["fully_forward1"], 0, sizeof(cl_mem), &d_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward1"], 1, sizeof(cl_mem), &fc1_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward1"], 2, sizeof(cl_mem), &e_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward1"], 3, sizeof(cl_uint), &a_height), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward1"], 4, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["fully_forward1"], 2, NULL, gws_ff1, lws_ff1, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // fully forward kernel 2, ok, so this is a different one, jk it's not
    // well, only if you think not including relu is different, then it is different
    q = 0;
    size_t lws_ff2[] = {TILE_SIZE, TILE_SIZE, 1};
    clFinish(queues[QUEUE_IDX_FC2]);
    for (unsigned int start = 0; start < e_dims[0]; start += batch_num) {
        const unsigned int a_height = min(batch_num, e_dims[0] - start);

        size_t gws_ff2[] = {((static_cast<unsigned int>(ceil(a_height / (float)TILE_SIZE)))*lws_ff2[0]), 1*lws_ff2[1], 1*lws_ff2[2]};

        // setting arguments and calling fully_forward1 kernel
        checkErr(clSetKernelArg(kernels["fully_forward2"], 0, sizeof(cl_mem), &e_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward2"], 1, sizeof(cl_mem), &fc2_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward2"], 2, sizeof(cl_mem), &f_device), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward2"], 3, sizeof(cl_uint), &a_height), __LINE__);
        checkErr(clSetKernelArg(kernels["fully_forward2"], 4, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["fully_forward2"], 2, NULL, gws_ff2, lws_ff2, 0, NULL, NULL), __LINE__);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // arg max kernel, search for the largest number...'s index in the array of arrays, the index is the index of the sub array (between 0 - 9)
    q = 0;
    size_t lws_argmax[] = {BLOCK_SIZE};
    for (unsigned int start = 0; start < f_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, f_dims[0] - start);

        size_t gws_argmax[] = {static_cast<unsigned int>(ceil(todo_count / (float)BLOCK_SIZE))*lws_argmax[0]};

        // setting arguments and calling fully_forward1 kernel
        checkErr(clSetKernelArg(kernels["arg_max"], 0, sizeof(cl_mem), &f_device), __LINE__);
        checkErr(clSetKernelArg(kernels["arg_max"], 1, sizeof(cl_mem), &output_device), __LINE__);
        checkErr(clSetKernelArg(kernels["arg_max"], 2, sizeof(cl_uint), &f_len), __LINE__);
        checkErr(clSetKernelArg(kernels["arg_max"], 3, sizeof(cl_uint), &start), __LINE__);
        checkErr(clEnqueueNDRangeKernel(queues[q], kernels["arg_max"], 1, NULL, gws_argmax, lws_argmax, 0, NULL, NULL), __LINE__);

        // copy final results back to host output buffer
        clEnqueueReadBuffer(queues[q], output_device, CL_FALSE, start, todo_count * sizeof(int),
                (void*)(output + start), 0, NULL, NULL);

        q = (q + 1) % NUM_CMD_QUEUES;
    }

    // wait for all of the queues to finish
    for(int i = 0; i < NUM_CMD_QUEUES; i++)
        checkErr(clFinish(queues[i]), __LINE__);
}

/*
    main(unsigned int argc, char **argv)
    DESCRIPTION:
        The main function

    INPUT:
        argc - number of args
        argv - pointer to args
*/
int main(int argc, char **argv) {

    if (argc != 4) {
        std::cerr << "This program performs the forward opertion step for Convolutional Neural Network(CNN). Sample usage: \n"
            << argv[0] << " test10.hdf5 model.hdf5 10\n";
        return -1;
    }

    initializeOpenCLParameters();

    FLAGS_testdata = std::string(argv[1]);
    FLAGS_model = std::string(argv[2]);
    FLAGS_batch_size = atoi(argv[3]);

    // set all the dimensions
    ref_dims[0] = FLAGS_batch_size;
    input_dims[0] = FLAGS_batch_size;
    input_unroll_dims[0] = FLAGS_batch_size;
    a_dims[0] = FLAGS_batch_size;
    b_dims[0] = FLAGS_batch_size;
    b_unroll_dims[0] = FLAGS_batch_size;
    c_dims[0] = FLAGS_batch_size;
    d_dims[0] = FLAGS_batch_size;
    d_dims2[0] = FLAGS_batch_size;
    e_dims[0] = FLAGS_batch_size;
    f_dims[0] = FLAGS_batch_size;

    // calculate the length of all the arrays
    conv1_len = flattened_length(conv1_dims);
    conv2_len = flattened_length(conv2_dims);
    fc1_len = flattened_length(fc1_dims);
    fc2_len = flattened_length(fc2_dims);
    input_len = flattened_length(input_dims);
    input_unroll_len = flattened_length(input_unroll_dims);
    a_len = flattened_length(a_dims);
    b_len = flattened_length(b_dims);
    b_unroll_len = flattened_length(b_unroll_dims);
    c_len = flattened_length(c_dims);
    d_len = flattened_length(d_dims);
    e_len = flattened_length(e_dims);
    f_len = flattened_length(f_dims);
    output_len = input_dims[0];

    // Create buffers for everything...
    conv1_device_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * conv1_len, NULL, NULL);
    conv2_device_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * conv2_len, NULL, NULL);
    fc1_device_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * fc1_len, NULL, NULL);
    conv1_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * conv1_len, NULL, NULL);
    conv2_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * conv2_len, NULL, NULL);
    fc1_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * fc1_len, NULL, NULL);
    fc2_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * fc2_len, NULL, NULL);
    input_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * input_len, NULL, NULL);
    input_unroll_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * input_unroll_len, NULL, NULL);
    a_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * a_len, NULL, NULL);
    b_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * b_len, NULL, NULL);
    b_unroll_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * b_unroll_len, NULL, NULL);
    c_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c_len, NULL, NULL);
    d_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * d_len, NULL, NULL);
    e_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * e_len, NULL, NULL);
    f_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * f_len, NULL, NULL);
    output_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * output_len, NULL, NULL);

    float *all_memory_host, *input, *conv1, *conv2, *fc1, *fc2;
    unsigned int *output;

    // malloc space for everything in host
    all_memory_host = (float *) malloc((input_len + conv1_len + conv2_len + fc1_len + fc2_len) * sizeof(float) + output_len * sizeof(int));
    input = &all_memory_host[0];
    conv1 = &input[input_len];
    conv2 = &conv1[conv1_len];
    fc1 = &conv2[conv2_len];
    fc2 = &fc1[fc1_len];
    output = (unsigned int *)&fc2[fc2_len];
    float *ref = allocate<float>(ref_dims);

    // Load data into input and ref
    load_data(input, ref);

    // Load model
    load_model(conv1, conv2, fc1, fc2);

    // do its thing
    const auto start = now();
    forward_operation(input, conv1, conv2, fc1, fc2, output);
    const auto end = now();

    // get elapsed time in milliseconds
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Get reference
    unsigned int *ground_truth = allocate<unsigned int>(FLAGS_batch_size);
    get_ground_truth(ref, ground_truth);

    // Calculate correctness
    unsigned int num_correct = 0;
    for (unsigned int i = 0; i < FLAGS_batch_size; i++) {
        if (output[i] == ground_truth[i]) {
            num_correct++;
        }
    }

    // prunsigned int the time and correctness
    std::cout << "Done with " << FLAGS_batch_size << " queries in "
        << "elapsed = " << elapsed << " milliseconds. Correctness: "
        << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

    // Clean up and free all resources
    cleanUp();
    free(all_memory_host);

    return 0;
}
