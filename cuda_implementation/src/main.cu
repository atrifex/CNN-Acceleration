/*
 * Copyright 2016 Fei Deng
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <time.h>
#include <valarray>
#include <string>
#include <hdf5.h>
#include <math.h>

#include "range.hpp"
#include "utils.hpp"


#define HALF_TILE_SIZE              16
#define TILE_SIZE                   32
#define DOUBLE_TILE_SIZE            64
#define BLOCK_SIZE                  512
#define POOL_SIZE                   2
#define AVG_COUNT                   (POOL_SIZE * POOL_SIZE)

#define NUM_DIGITS                  10
#define NUM_STREAMS                 16
#define STREAM_IDX_CONV1            (NUM_STREAMS - 1)
#define STREAM_IDX_CONV2            (NUM_STREAMS - 2)
#define STREAM_IDX_FC1              (NUM_STREAMS - 3)
#define STREAM_IDX_FC2              (NUM_STREAMS - 4)
#define MAX_THREADS_PER_BLOCK       1024

#define CONV_ROWS                   5
#define CONV_COLS                   5
#define CONV_NUM_ELEMENTS           (CONV_ROWS * CONV_COLS)
#define CONV1_INPUT_CHANNELS        1
#define CONV1_OUTPUT_CHANNELS       32
#define CONV1_NUM_ELEMENTS_IN       (CONV1_INPUT_CHANNELS * CONV_NUM_ELEMENTS)
#define CONV2_INPUT_CHANNELS        32
#define CONV2_OUTPUT_CHANNELS       64
#define CONV2_NUM_ELEMENTS_IN       (CONV2_INPUT_CHANNELS * CONV_NUM_ELEMENTS)

#define FC1_ROWS                    1024
#define FC1_COLS                    128
#define FC2_ROWS                    FC1_COLS
#define FC2_COLS                    NUM_DIGITS

#define INPUT_ROWS                  28
#define INPUT_COLS                  28
#define INPUT_CHANNELS              1
#define INPUT_NUM_ELEMENTS          (INPUT_ROWS * INPUT_COLS)

#define A_ROWS                      (INPUT_ROWS - CONV_ROWS + 1)
#define A_COLS                      (INPUT_COLS - CONV_COLS + 1)
#define A_NUM_ELEMENTS              (A_ROWS * A_COLS)
#define B_ROWS                      (A_ROWS / 2)
#define B_COLS                      (A_COLS / 2)
#define B_NUM_ELEMENTS              (B_ROWS * B_COLS)
#define C_ROWS                      (B_ROWS - CONV_ROWS + 1)
#define C_COLS                      (B_COLS - CONV_COLS + 1)
#define C_NUM_ELEMENTS              (C_ROWS * C_COLS)
#define D_ROWS                      (C_ROWS / 2)
#define D_COLS                      (C_COLS / 2)
#define D_NUM_ELEMENTS              (D_ROWS * D_COLS)

#define AVG1_A_NUM_ELEMENTS_OUT     (CONV1_OUTPUT_CHANNELS * A_NUM_ELEMENTS)
#define AVG1_B_NUM_ELEMENTS_OUT     (CONV1_OUTPUT_CHANNELS * B_NUM_ELEMENTS)
#define AVG2_C_NUM_ELEMENTS_OUT     (CONV2_OUTPUT_CHANNELS * C_NUM_ELEMENTS)
#define AVG2_D_NUM_ELEMENTS_OUT     (CONV2_OUTPUT_CHANNELS * D_NUM_ELEMENTS)

#define BATCH_NUM_PER_STREAM        1000
#define BATCH_NUM_FACTOR            32
#define CONSTANT_MEM_SIZE           (64 * 1024)

#define MATRIX_MUL1_BLOCKS_PER_NUM  6
#define MATRIX_MUL1_COLS_PER_BLOCK  (A_NUM_ELEMENTS / MATRIX_MUL1_BLOCKS_PER_NUM)
#define MATRIX_MUL1_SKIP1           (CONV_NUM_ELEMENTS * A_NUM_ELEMENTS)
#define MATRIX_MUL1_SKIP2           (CONV1_OUTPUT_CHANNELS * A_NUM_ELEMENTS)
#define MATRIX_MUL2_NUMS_PER_BLOCK  2
#define UNROLL2_LAYERS              (BLOCK_SIZE / C_NUM_ELEMENTS)
#define FULLY_FORWARD1_TILE_NUM     (FC1_ROWS / TILE_SIZE)

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
float *all_memory_device;
float *conv1_device_;
float *conv2_device_;
float *fc1_device_;

float *conv1_device;
float *conv2_device;
float *fc1_device;
float *fc2_device;

float *input_device;
float *input_unroll_device;

float *a_device;
float *b_device;
float *b_unroll_device;
float *c_device;
float *d_device;
float *e_device;
float *f_device;

unsigned int *output_device;

// CUDA streams
cudaStream_t streams[NUM_STREAMS];

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
    transform_conv1(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv1 to the normal form.
    INPUT:
        input - pointer to the old conv1 array
        output - pointer to the new conv1 array
*/
__global__ void transform_conv1(const float *input, float *output) {

    // old dimmensions are (row=5, col=5, input_channel=1, output_channel=32)
    // new dimmensions are (output_channel=32, input_channel=1, row=5, col=5)
    output[threadIdx.z * CONV_ROWS * CONV_COLS + threadIdx.y * CONV_COLS + threadIdx.x] =
        input[threadIdx.y * CONV_COLS * CONV1_OUTPUT_CHANNELS + threadIdx.x * CONV1_OUTPUT_CHANNELS + threadIdx.z];
}

/*
    transform_conv2(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv2 to the normal form.
    INPUT:
        input - pointer to the old conv2 array
        output - pointer to the new conv2 array
*/
__global__ void transform_conv2(const float *input, float *output) {

    // old dimmensions are (row=5, col=5, input_channel=32, output_channel=64)
    // new dimmensions are (output_channel=64, input_channel=32, row=5, col=5)
    output[blockIdx.x * CONV2_INPUT_CHANNELS * CONV_ROWS * CONV_COLS +
            threadIdx.z * CONV_ROWS * CONV_COLS + threadIdx.y * CONV_COLS + threadIdx.x] =
        input[threadIdx.y * CONV_COLS * CONV2_INPUT_CHANNELS * CONV2_OUTPUT_CHANNELS +
            threadIdx.x * CONV2_INPUT_CHANNELS * CONV2_OUTPUT_CHANNELS + threadIdx.z * CONV2_OUTPUT_CHANNELS + blockIdx.x];
}

/*
    transform_fc1(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv2 to the normal form.
    INPUT:
        input - pointer to the old fc1 array
        output - pointer to the new fc1 array
*/
__global__ void transform_fc1(const float *input, float *output) {

    // old dimmensions are (row=1024, col=128), the sub dimmensions for row are (sub_row=4, sub_col=4, channel=64)
    // old dimmensions are (row=1024, col=128), the sub dimmensions for row are (channel=64, sub_row=4, sub_col=4)
    output[threadIdx.z * D_ROWS * D_COLS * FC1_COLS + threadIdx.y * D_COLS * FC1_COLS +
            threadIdx.x * FC1_COLS + blockIdx.x] =
        input[threadIdx.y * D_COLS * CONV2_OUTPUT_CHANNELS * FC1_COLS +
            threadIdx.x * CONV2_OUTPUT_CHANNELS * FC1_COLS + threadIdx.z * FC1_COLS + blockIdx.x];
}

/*
    unroll1(const float *x, float *x_unroll, const unsigned int start)
    DESCRIPTION:
        Kernel function for unrolling input data from test file.
    INPUT:
        x - pointer to the input array
        x_unroll - pointer to the input_unroll array
        start - the first channel's index of the current batch
*/
__global__ void unroll1(const float *x, float *x_unroll, const unsigned int start) {

    // 5 * 5 from filter size (per channel)
    const unsigned int x_unroll_height = CONV_NUM_ELEMENTS;

    // 24 * 24 from new size after multiplication
    const unsigned int x_unroll_width = A_NUM_ELEMENTS;

    // every block unrolls 1 channel
    // channel_idx is the universal index of channels for all input data
    const unsigned int channel_idx = start + blockIdx.x;

    // the amount of addresses in one channel
    const unsigned int address_per_channel = x_unroll_height * x_unroll_width;

    // load a channel of inputs into shared memory
    __shared__ float x_shared[INPUT_NUM_ELEMENTS];
    const unsigned int x_base = channel_idx * INPUT_NUM_ELEMENTS;
    for (unsigned int i = threadIdx.x; i < INPUT_NUM_ELEMENTS; i += blockDim.x) {
        x_shared[i] = x[x_base + i];
    }
    __syncthreads();

    unsigned int t = threadIdx.x;
    while (t < x_unroll_width) {

        // starting row number in the current channel
        const unsigned int x_row_base = t / A_ROWS;

        // starting col number in the current channel
        const unsigned int x_col_base = t % A_COLS;

        // offset to the first address of the col after unrolling
        unsigned int x_unroll_offset = channel_idx * address_per_channel + t;

        // unroll the matrix
        for (unsigned int i = 0; i < CONV_ROWS; i++) {
            const unsigned int x_start = (x_row_base + i) * INPUT_COLS;
            for (unsigned int j = 0; j < CONV_COLS; j++) {
                x_unroll[x_unroll_offset] = x_shared[x_start + (x_col_base + j)];
                x_unroll_offset += x_unroll_width;
            }
        }

        // every block has only 512 threads for better GPU utilization
        // but width of unrolled matrix is 576, so the first 64 needs to do another round
        t += BLOCK_SIZE;
    }
}

/*
    unroll2(const float *x, float *x_unroll, const unsigned int start)
    DESCRIPTION:
        Kernel function for unrolling intermediate data.
    INPUT:
        x - pointer to the b array
        x_unroll - pointer to the b_unroll array
        start - the first channel's index of the current batch
*/
__global__ void unroll2(const float *x, float *x_unroll, const unsigned int start) {

    // 5 * 5 from filter size (per channel)
    const unsigned int x_unroll_height = CONV_NUM_ELEMENTS;

    // 8 * 8 from new size after multiplication
    const unsigned int x_unroll_width = C_NUM_ELEMENTS;

    // every block unrolls 8 channelS
    // channel_idx is the universal index of channels for all input data
    const unsigned int channel_idx = start + blockIdx.x * blockDim.y + threadIdx.y;

    // the amount of addresses in one channel
    const unsigned int address_per_channel = x_unroll_height * x_unroll_width;

    // load a channel of inputs into shared memory
    __shared__ float x_shared[UNROLL2_LAYERS][B_NUM_ELEMENTS];
    const unsigned int x_base = channel_idx * B_NUM_ELEMENTS;
    for (unsigned int i = threadIdx.x; i < B_NUM_ELEMENTS; i += blockDim.x) {
        x_shared[threadIdx.y][i] = x[x_base + i];
    }
    __syncthreads();

    const unsigned int t = threadIdx.x;

    // starting row number in the current channel
    const unsigned int x_row_base = t / C_ROWS;

    // starting col number in the current channel
    const unsigned int x_col_base = t % C_COLS;

    // offset to the first address of the col after unrolling
    unsigned int x_unroll_offset = channel_idx * address_per_channel + t;

    // unroll the matrix
    for (unsigned int i = 0; i < CONV_ROWS; i++) {
        const unsigned int x_start = (x_row_base + i) * B_COLS;
        for (unsigned int j = 0; j < CONV_COLS; j++) {
            x_unroll[x_unroll_offset] = x_shared[threadIdx.y][x_start + (x_col_base + j)];
            x_unroll_offset += x_unroll_width;
        }
    }
}

/*
    matrix_multiplication1(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start)
    DESCRIPTION:
        Kernel function for tile based matrix multiplication.

        Every block has 16 * 16 threads, but tiles are still 32 * 32,
        and each block deals with 1/6 of a number.

        This means:
            1. each thread loads 4 inputs from matrix a
            2. each thread loads 12 inputs from matrix b
            3. each thread writes 12 outputs to matrix c
    INPUT:
        matrix_b - pointer to the input_unroll array
        matrix_c - pointer to the a array
        start - the first number's index of the current batch
*/
__global__ void matrix_multiplication1(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start) {

    // matrix_a is not needed here since it's in constant memory
    const unsigned int a_width = CONV1_NUM_ELEMENTS_IN;
    const unsigned int b_height = CONV_NUM_ELEMENTS;
    const unsigned int b_width = A_NUM_ELEMENTS;
    const unsigned int c_height = CONV1_OUTPUT_CHANNELS;
    const unsigned int c_width = b_width;

    // each block deals with 1/6 of a number
    // block index in x direction is for one of these 6 blocks
    // block index in y direction is for number indices
    const unsigned int num_idx = blockIdx.y;

    const unsigned int matrix_c_row1 = threadIdx.y;
    const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;

    const unsigned int matrix_c_col1 = blockIdx.x * MATRIX_MUL1_COLS_PER_BLOCK + threadIdx.x;
    const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col5 = matrix_c_col4 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col6 = matrix_c_col5 + HALF_TILE_SIZE;

    // size of unrolled input per channel is 25 * 576, each block is 16 * 16
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b_1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b_2[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b_3[TILE_SIZE][TILE_SIZE];

    const unsigned int tile_row1 = threadIdx.y;
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_col1 = threadIdx.x;
    const unsigned int tile_col2 = tile_col1 + HALF_TILE_SIZE;

    // arrays should be allocated in registers but the speed was a lot
    // slower than using just local variables which also use registers
    float c_01 = 0; float c_02 = 0; float c_03 = 0; float c_04 = 0; float c_05 = 0; float c_06 = 0;
    float c_07 = 0; float c_08 = 0; float c_09 = 0; float c_10 = 0; float c_11 = 0; float c_12 = 0;

    tile_a[tile_row1][tile_col1] = matrix_a[matrix_c_row1 * a_width + tile_col1];
    tile_a[tile_row2][tile_col1] = matrix_a[matrix_c_row2 * a_width + tile_col1];
    if (tile_col2 < a_width) {
        tile_a[tile_row1][tile_col2] = matrix_a[matrix_c_row1 * a_width + tile_col2];
        tile_a[tile_row2][tile_col2] = matrix_a[matrix_c_row2 * a_width + tile_col2];
    }
    else {
        tile_a[tile_row1][tile_col2] = 0;
        tile_a[tile_row2][tile_col2] = 0;
    }

    // using registers to store common indecies
    const unsigned int matrix_b_start = (start + num_idx) * b_height * b_width;
    const unsigned int matrix_b_row1 = tile_row1 * b_width;

    // load the tile for matrix b (unrolled input array)
    tile_b_1[tile_row1][tile_col1] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col1];
    tile_b_1[tile_row1][tile_col2] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col2];
    tile_b_2[tile_row1][tile_col1] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col3];
    tile_b_2[tile_row1][tile_col2] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col4];
    tile_b_3[tile_row1][tile_col1] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col5];
    tile_b_3[tile_row1][tile_col2] = matrix_b[matrix_b_start + matrix_b_row1 + matrix_c_col6];

    if (tile_row2 < b_height) {
        const unsigned int matrix_b_row2 = tile_row2 * b_width;
        tile_b_1[tile_row2][tile_col1] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col1];
        tile_b_1[tile_row2][tile_col2] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col2];
        tile_b_2[tile_row2][tile_col1] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col3];
        tile_b_2[tile_row2][tile_col2] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col4];
        tile_b_3[tile_row2][tile_col1] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col5];
        tile_b_3[tile_row2][tile_col2] = matrix_b[matrix_b_start + matrix_b_row2 + matrix_c_col6];
    }
    else {
        tile_b_1[tile_row2][tile_col1] = 0;
        tile_b_1[tile_row2][tile_col2] = 0;
        tile_b_2[tile_row2][tile_col1] = 0;
        tile_b_2[tile_row2][tile_col2] = 0;
        tile_b_3[tile_row2][tile_col1] = 0;
        tile_b_3[tile_row2][tile_col2] = 0;
    }

    __syncthreads();
    for (unsigned int k = 0; k < CONV_NUM_ELEMENTS; k++) {
        const float a_1 = tile_a[tile_row1][k];
        const float a_2 = tile_a[tile_row2][k];
        const float b_1 = tile_b_1[k][tile_col1];
        const float b_2 = tile_b_1[k][tile_col2];
        const float b_3 = tile_b_2[k][tile_col1];
        const float b_4 = tile_b_2[k][tile_col2];
        const float b_5 = tile_b_3[k][tile_col1];
        const float b_6 = tile_b_3[k][tile_col2];

        c_01 += a_1 * b_1; c_02 += a_1 * b_2; c_03 += a_1 * b_3;
        c_04 += a_1 * b_4; c_05 += a_1 * b_5; c_06 += a_1 * b_6;

        c_07 += a_2 * b_1; c_08 += a_2 * b_2; c_09 += a_2 * b_3;
        c_10 += a_2 * b_4; c_11 += a_2 * b_5; c_12 += a_2 * b_6;
    }

    // relu is included in here
    const unsigned int matrix_c_num_start = (start + num_idx) * c_height * c_width;
    const unsigned int matrix_c_row1_start = matrix_c_row1 * c_width;
    const unsigned int matrix_c_row2_start = matrix_c_row2 * c_width;

    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col1] = (c_01 <= 0) ? 0 : c_01;
    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col2] = (c_02 <= 0) ? 0 : c_02;
    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col3] = (c_03 <= 0) ? 0 : c_03;
    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col4] = (c_04 <= 0) ? 0 : c_04;
    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col5] = (c_05 <= 0) ? 0 : c_05;
    matrix_c[matrix_c_num_start + matrix_c_row1_start + matrix_c_col6] = (c_06 <= 0) ? 0 : c_06;

    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col1] = (c_07 <= 0) ? 0 : c_07;
    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col2] = (c_08 <= 0) ? 0 : c_08;
    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col3] = (c_09 <= 0) ? 0 : c_09;
    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col4] = (c_10 <= 0) ? 0 : c_10;
    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col5] = (c_11 <= 0) ? 0 : c_11;
    matrix_c[matrix_c_num_start + matrix_c_row2_start + matrix_c_col6] = (c_12 <= 0) ? 0 : c_12;
}

/*
    matrix_multiplication2(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start)
    DESCRIPTION:
        Kernel function for tile based matrix multiplication.

        Every block has 16 * 16 threads, but tiles are 64 * 32 for matrix a, and 32 * 64 for matrix b.
        Also, each block deals with 2 numbers.

        This means:
            1. each thread loads 8 inputs from matrix a
            2. each thread loads 16 inputs from matrix b
            3. each thread writes 32 outputs to matrix c
    INPUT:
        matrix_a - pointer to the conv2 array
        matrix_b - pointer to the b_unroll array
        matrix_c - pointer to the c array
        start - the first number's index of the current batch
*/
__global__ void matrix_multiplication2(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start) {

    const unsigned int a_width = CONV2_NUM_ELEMENTS_IN;
    const unsigned int b_height = a_width;
    const unsigned int b_width = C_NUM_ELEMENTS;
    const unsigned int c_height = CONV2_OUTPUT_CHANNELS;
    const unsigned int c_width = b_width;

    // const unsigned int matrix_c_row1 = threadIdx.y;
    // const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_row3 = matrix_c_row2 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_row4 = matrix_c_row3 + HALF_TILE_SIZE;

    // const unsigned int matrix_c_col1 = threadIdx.x;
    // const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;

    // every block deals with 2 numbers
    const unsigned int num_idx = blockIdx.x * MATRIX_MUL2_NUMS_PER_BLOCK;

    // each block is 16 * 16,
    // we load a tile of 64 * 32 from matrix a, since matrix a is 64 * 800
    // we also load a tile of 32 * 64 from matrix b, since matrix b is 800 * 64 per number
    __shared__ float tile_a[DOUBLE_TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b1[TILE_SIZE][DOUBLE_TILE_SIZE];
    __shared__ float tile_b2[TILE_SIZE][DOUBLE_TILE_SIZE];

    const unsigned int tile_row1 = threadIdx.y;
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_row3 = tile_row2 + HALF_TILE_SIZE;
    const unsigned int tile_row4 = tile_row3 + HALF_TILE_SIZE;

    const unsigned int tile_col1 = threadIdx.x;
    const unsigned int tile_col2 = tile_col1 + HALF_TILE_SIZE;
    const unsigned int tile_col3 = tile_col2 + HALF_TILE_SIZE;
    const unsigned int tile_col4 = tile_col3 + HALF_TILE_SIZE;

    // arrays should be allocated in registers but the speed was a lot
    // slower than using just local variables which also use registers
    float c1_01 = 0; float c1_02 = 0; float c1_03 = 0; float c1_04 = 0;
    float c1_05 = 0; float c1_06 = 0; float c1_07 = 0; float c1_08 = 0;
    float c1_09 = 0; float c1_10 = 0; float c1_11 = 0; float c1_12 = 0;
    float c1_13 = 0; float c1_14 = 0; float c1_15 = 0; float c1_16 = 0;

    float c2_01 = 0; float c2_02 = 0; float c2_03 = 0; float c2_04 = 0;
    float c2_05 = 0; float c2_06 = 0; float c2_07 = 0; float c2_08 = 0;
    float c2_09 = 0; float c2_10 = 0; float c2_11 = 0; float c2_12 = 0;
    float c2_13 = 0; float c2_14 = 0; float c2_15 = 0; float c2_16 = 0;

    // tile number = a_width / CONV2_INPUT_CHANNELS = CONV_ROWS * CONV_COLS
    for (unsigned int tile = 0; tile < CONV_NUM_ELEMENTS; tile++) {

        // use registers to store common indices to speed up
        const unsigned int matrix_a_col1 = tile * TILE_SIZE + tile_col1;
        const unsigned int matrix_a_col2 = tile * TILE_SIZE + tile_col2;
        const unsigned int matrix_b_row1 = (tile * TILE_SIZE + tile_row1) * b_width;
        const unsigned int matrix_b_row2 = (tile * TILE_SIZE + tile_row2) * b_width;
        const unsigned int matrix_b_num1_start = (start + num_idx) * b_height * b_width;
        const unsigned int matrix_b_num2_start = (start + num_idx + 1) * b_height * b_width;

        // don't need to check if row or col is in bound since everything here is divisible by 32

        // all the tile_row# and tile_col# on the right side of the equal sign here
        // used to be matrix_c_row# and matrix_c_col#
        // but we are running out of registers, which is a good thing, and since they have the
        // same value as tile_row# and tile_col#, replacing them has no harm
        tile_a[tile_row1][tile_col1] = matrix_a[tile_row1 * a_width + matrix_a_col1];
        tile_a[tile_row2][tile_col1] = matrix_a[tile_row2 * a_width + matrix_a_col1];
        tile_a[tile_row3][tile_col1] = matrix_a[tile_row3 * a_width + matrix_a_col1];
        tile_a[tile_row4][tile_col1] = matrix_a[tile_row4 * a_width + matrix_a_col1];
        tile_a[tile_row1][tile_col2] = matrix_a[tile_row1 * a_width + matrix_a_col2];
        tile_a[tile_row2][tile_col2] = matrix_a[tile_row2 * a_width + matrix_a_col2];
        tile_a[tile_row3][tile_col2] = matrix_a[tile_row3 * a_width + matrix_a_col2];
        tile_a[tile_row4][tile_col2] = matrix_a[tile_row4 * a_width + matrix_a_col2];

        tile_b1[tile_row1][tile_col1] = matrix_b[matrix_b_num1_start + matrix_b_row1 + tile_col1];
        tile_b1[tile_row2][tile_col1] = matrix_b[matrix_b_num1_start + matrix_b_row2 + tile_col1];
        tile_b1[tile_row1][tile_col2] = matrix_b[matrix_b_num1_start + matrix_b_row1 + tile_col2];
        tile_b1[tile_row2][tile_col2] = matrix_b[matrix_b_num1_start + matrix_b_row2 + tile_col2];
        tile_b1[tile_row1][tile_col3] = matrix_b[matrix_b_num1_start + matrix_b_row1 + tile_col3];
        tile_b1[tile_row2][tile_col3] = matrix_b[matrix_b_num1_start + matrix_b_row2 + tile_col3];
        tile_b1[tile_row1][tile_col4] = matrix_b[matrix_b_num1_start + matrix_b_row1 + tile_col4];
        tile_b1[tile_row2][tile_col4] = matrix_b[matrix_b_num1_start + matrix_b_row2 + tile_col4];

        tile_b2[tile_row1][tile_col1] = matrix_b[matrix_b_num2_start + matrix_b_row1 + tile_col1];
        tile_b2[tile_row2][tile_col1] = matrix_b[matrix_b_num2_start + matrix_b_row2 + tile_col1];
        tile_b2[tile_row1][tile_col2] = matrix_b[matrix_b_num2_start + matrix_b_row1 + tile_col2];
        tile_b2[tile_row2][tile_col2] = matrix_b[matrix_b_num2_start + matrix_b_row2 + tile_col2];
        tile_b2[tile_row1][tile_col3] = matrix_b[matrix_b_num2_start + matrix_b_row1 + tile_col3];
        tile_b2[tile_row2][tile_col3] = matrix_b[matrix_b_num2_start + matrix_b_row2 + tile_col3];
        tile_b2[tile_row1][tile_col4] = matrix_b[matrix_b_num2_start + matrix_b_row1 + tile_col4];
        tile_b2[tile_row2][tile_col4] = matrix_b[matrix_b_num2_start + matrix_b_row2 + tile_col4];

        __syncthreads();
        for (unsigned int k = 0; k < TILE_SIZE; k++) {

            // use registers to store values from tile a and b, since we are using them a lot
            // in here it provides a speed boost
            const float a_01 = tile_a[tile_row1][k];
            const float a_02 = tile_a[tile_row2][k];
            const float a_03 = tile_a[tile_row3][k];
            const float a_04 = tile_a[tile_row4][k];
            const float b1_01 = tile_b1[k][tile_col1];
            const float b1_02 = tile_b1[k][tile_col2];
            const float b1_03 = tile_b1[k][tile_col3];
            const float b1_04 = tile_b1[k][tile_col4];
            const float b2_01 = tile_b2[k][tile_col1];
            const float b2_02 = tile_b2[k][tile_col2];
            const float b2_03 = tile_b2[k][tile_col3];
            const float b2_04 = tile_b2[k][tile_col4];

            c1_01 += a_01 * b1_01; c1_02 += a_01 * b1_02; c1_03 += a_01 * b1_03; c1_04 += a_01 * b1_04;
            c1_05 += a_02 * b1_01; c1_06 += a_02 * b1_02; c1_07 += a_02 * b1_03; c1_08 += a_02 * b1_04;
            c1_09 += a_03 * b1_01; c1_10 += a_03 * b1_02; c1_11 += a_03 * b1_03; c1_12 += a_03 * b1_04;
            c1_13 += a_04 * b1_01; c1_14 += a_04 * b1_02; c1_15 += a_04 * b1_03; c1_16 += a_04 * b1_04;

            c2_01 += a_01 * b2_01; c2_02 += a_01 * b2_02; c2_03 += a_01 * b2_03; c2_04 += a_01 * b2_04;
            c2_05 += a_02 * b2_01; c2_06 += a_02 * b2_02; c2_07 += a_02 * b2_03; c2_08 += a_02 * b2_04;
            c2_09 += a_03 * b2_01; c2_10 += a_03 * b2_02; c2_11 += a_03 * b2_03; c2_12 += a_03 * b2_04;
            c2_13 += a_04 * b2_01; c2_14 += a_04 * b2_02; c2_15 += a_04 * b2_03; c2_16 += a_04 * b2_04;
        }
        __syncthreads();
    }

    // same as aboe, all tile_row/col used to be matrix_c_row/col
    const unsigned int matrix_c_num1_start = (start + num_idx) * c_height * c_width;
    const unsigned int matrix_c_num2_start = (start + num_idx + 1) * c_height * c_width;
    const unsigned int matrix_c_row1_start = tile_row1 * c_width;
    const unsigned int matrix_c_row2_start = tile_row2 * c_width;
    const unsigned int matrix_c_row3_start = tile_row3 * c_width;
    const unsigned int matrix_c_row4_start = tile_row4 * c_width;

    // relu is included in here
    matrix_c[matrix_c_num1_start + matrix_c_row1_start + tile_col1] = (c1_01 <= 0) ? 0 : c1_01;
    matrix_c[matrix_c_num1_start + matrix_c_row1_start + tile_col2] = (c1_02 <= 0) ? 0 : c1_02;
    matrix_c[matrix_c_num1_start + matrix_c_row1_start + tile_col3] = (c1_03 <= 0) ? 0 : c1_03;
    matrix_c[matrix_c_num1_start + matrix_c_row1_start + tile_col4] = (c1_04 <= 0) ? 0 : c1_04;
    matrix_c[matrix_c_num1_start + matrix_c_row2_start + tile_col1] = (c1_05 <= 0) ? 0 : c1_05;
    matrix_c[matrix_c_num1_start + matrix_c_row2_start + tile_col2] = (c1_06 <= 0) ? 0 : c1_06;
    matrix_c[matrix_c_num1_start + matrix_c_row2_start + tile_col3] = (c1_07 <= 0) ? 0 : c1_07;
    matrix_c[matrix_c_num1_start + matrix_c_row2_start + tile_col4] = (c1_08 <= 0) ? 0 : c1_08;
    matrix_c[matrix_c_num1_start + matrix_c_row3_start + tile_col1] = (c1_09 <= 0) ? 0 : c1_09;
    matrix_c[matrix_c_num1_start + matrix_c_row3_start + tile_col2] = (c1_10 <= 0) ? 0 : c1_10;
    matrix_c[matrix_c_num1_start + matrix_c_row3_start + tile_col3] = (c1_11 <= 0) ? 0 : c1_11;
    matrix_c[matrix_c_num1_start + matrix_c_row3_start + tile_col4] = (c1_12 <= 0) ? 0 : c1_12;
    matrix_c[matrix_c_num1_start + matrix_c_row4_start + tile_col1] = (c1_13 <= 0) ? 0 : c1_13;
    matrix_c[matrix_c_num1_start + matrix_c_row4_start + tile_col2] = (c1_14 <= 0) ? 0 : c1_14;
    matrix_c[matrix_c_num1_start + matrix_c_row4_start + tile_col3] = (c1_15 <= 0) ? 0 : c1_15;
    matrix_c[matrix_c_num1_start + matrix_c_row4_start + tile_col4] = (c1_16 <= 0) ? 0 : c1_16;

    matrix_c[matrix_c_num2_start + matrix_c_row1_start + tile_col1] = (c2_01 <= 0) ? 0 : c2_01;
    matrix_c[matrix_c_num2_start + matrix_c_row1_start + tile_col2] = (c2_02 <= 0) ? 0 : c2_02;
    matrix_c[matrix_c_num2_start + matrix_c_row1_start + tile_col3] = (c2_03 <= 0) ? 0 : c2_03;
    matrix_c[matrix_c_num2_start + matrix_c_row1_start + tile_col4] = (c2_04 <= 0) ? 0 : c2_04;
    matrix_c[matrix_c_num2_start + matrix_c_row2_start + tile_col1] = (c2_05 <= 0) ? 0 : c2_05;
    matrix_c[matrix_c_num2_start + matrix_c_row2_start + tile_col2] = (c2_06 <= 0) ? 0 : c2_06;
    matrix_c[matrix_c_num2_start + matrix_c_row2_start + tile_col3] = (c2_07 <= 0) ? 0 : c2_07;
    matrix_c[matrix_c_num2_start + matrix_c_row2_start + tile_col4] = (c2_08 <= 0) ? 0 : c2_08;
    matrix_c[matrix_c_num2_start + matrix_c_row3_start + tile_col1] = (c2_09 <= 0) ? 0 : c2_09;
    matrix_c[matrix_c_num2_start + matrix_c_row3_start + tile_col2] = (c2_10 <= 0) ? 0 : c2_10;
    matrix_c[matrix_c_num2_start + matrix_c_row3_start + tile_col3] = (c2_11 <= 0) ? 0 : c2_11;
    matrix_c[matrix_c_num2_start + matrix_c_row3_start + tile_col4] = (c2_12 <= 0) ? 0 : c2_12;
    matrix_c[matrix_c_num2_start + matrix_c_row4_start + tile_col1] = (c2_13 <= 0) ? 0 : c2_13;
    matrix_c[matrix_c_num2_start + matrix_c_row4_start + tile_col2] = (c2_14 <= 0) ? 0 : c2_14;
    matrix_c[matrix_c_num2_start + matrix_c_row4_start + tile_col3] = (c2_15 <= 0) ? 0 : c2_15;
    matrix_c[matrix_c_num2_start + matrix_c_row4_start + tile_col4] = (c2_16 <= 0) ? 0 : c2_16;
}

/*
    average_pool1(const float *x, float *y, const unsigned int start, const unsigned int end)
    DESCRIPTION:
        Kernel function for averaging results of matrix multiplication.
    INPUT:
        x - pointer to the a array
        y - pointer to the b array
        start - the first number's index of the current batch
        end - the total count of numbers in the input data (each number consists of channels * row * col addresses)
*/
__global__ void average_pool1(const float *x, float *y, const unsigned int start, const unsigned int end) {

    const float count = AVG_COUNT;
    const unsigned int row = threadIdx.y;
    const unsigned int col = threadIdx.x;

    // the index of the number this thread is averaging
    const unsigned int num_idx = start + blockIdx.x * blockDim.z + threadIdx.z;

    if (num_idx < end) {

        // base address of x and y for the current number
        const unsigned int x_base = num_idx * AVG1_A_NUM_ELEMENTS_OUT;
        const unsigned int y_base = num_idx * AVG1_B_NUM_ELEMENTS_OUT;

        // loop through all channels and calculate the average
        for (unsigned int c = 0; c < CONV1_OUTPUT_CHANNELS; c++) {
            float output = 0;
            for (unsigned int p = 0; p < POOL_SIZE; p++) {
                for (unsigned int q = 0; q < POOL_SIZE; q++) {

                    // bit shift here since it's faster to do this than multiply
                    // shift by 1 is for multiply POOL_SIZE
                    unsigned int x_offset = x_base + c * A_NUM_ELEMENTS + ((row << 1) + p) * A_COLS + ((col << 1) + q);
                    output += x[x_offset];
                }
            }
            const unsigned int y_offset = y_base + c * B_NUM_ELEMENTS + row * B_COLS + col;
            y[y_offset] = output ? output / count : 0;
        }
    }
}

/*
    average_pool2(const float *x, float *y, const unsigned int start, const unsigned int end)
    DESCRIPTION:
        Kernel function for averaging results of matrix multiplication.
    INPUT:
        x - pointer to the c array
        y - pointer to the d array
        start - the first number index of the current batch
        end - the total count of numbers in the input data (each number consists of channels * row * col addresses)
*/
__global__ void average_pool2(const float *x, float *y, const unsigned int start, const unsigned int end) {

    const float count = AVG_COUNT;
    const unsigned int row = threadIdx.y;
    const unsigned int col = threadIdx.x;

    // the index of the number this thread is averaging
    const unsigned int num_idx = start + blockIdx.x * blockDim.z + threadIdx.z;

    if (num_idx < end) {

        // base address of x and y for the current number
        const unsigned int x_base = num_idx * AVG2_C_NUM_ELEMENTS_OUT;
        const unsigned int y_base = num_idx * AVG2_D_NUM_ELEMENTS_OUT;

        // loop through all channels and calculate the average
        for (unsigned int c = 0; c < CONV2_OUTPUT_CHANNELS; c++) {
            float output = 0;
            for (unsigned int p = 0; p < POOL_SIZE; p++) {
                for (unsigned int q = 0; q < POOL_SIZE; q++) {

                    // bit shift here since it's faster to do this than multiply
                    // shift by 1 is for multiply POOL_SIZE
                    // shift by 3 is for multiply C_COLS
                    unsigned int x_offset = x_base + c * C_NUM_ELEMENTS + (((row << 1) + p) << 3) + ((col << 1) + q);
                    output += x[x_offset];
                }
            }
            const unsigned int y_offset = y_base + c * D_NUM_ELEMENTS + row * D_COLS + col;
            y[y_offset] = output ? output / count : 0;
        }
    }
}

/*
    fully_forward1(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start)
    DESCRIPTION:
        Kernel function for tile based matrix multiplication specialized for fully forward.
    INPUT:
        matrix_a - pointer to the d array
        matrix_b - pointer to the fc1 array
        matrix_c - pointer to the e array
        a_height - the count of numbers of the current batch
        start - the first number's index of the current batch
*/
__global__ void fully_forward1(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int a_height, const unsigned int start) {

    const unsigned int a_width = FC1_ROWS;
    const unsigned int b_width = FC1_COLS;
    const unsigned int c_height = a_height;
    const unsigned int c_width = b_width;

    const unsigned int matrix_c_row1 = blockIdx.x * TILE_SIZE + threadIdx.y;
    const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;

    const unsigned int matrix_c_col1 = threadIdx.x;
    const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col5 = matrix_c_col4 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col6 = matrix_c_col5 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col7 = matrix_c_col6 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col8 = matrix_c_col7 + HALF_TILE_SIZE;

    // each block is 16 * 16
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b2[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b3[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b4[TILE_SIZE][TILE_SIZE];

    const unsigned int tile_row1 = threadIdx.y;
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_col1 = threadIdx.x;
    const unsigned int tile_col2 = tile_col1 + HALF_TILE_SIZE;

    float c_01 = 0; float c_02 = 0; float c_03 = 0; float c_04 = 0; float c_05 = 0; float c_06 = 0; float c_07 = 0; float c_08 = 0;
    float c_09 = 0; float c_10 = 0; float c_11 = 0; float c_12 = 0; float c_13 = 0; float c_14 = 0; float c_15 = 0; float c_16 = 0;

    // load the tiles and accumulate the product
    for (unsigned int tile = 0; tile < FULLY_FORWARD1_TILE_NUM; tile++) {
        if (matrix_c_row2 < a_height) {
            const unsigned int matrix_a_start = start * a_width;
            const unsigned int matrix_a_row1_start = matrix_c_row1 * a_width;
            const unsigned int matrix_a_row2_start = matrix_c_row2 * a_width;
            const unsigned int matrix_a_col_start = tile * TILE_SIZE;
            tile_a[tile_row1][tile_col1] = matrix_a[matrix_a_start + matrix_a_row1_start + matrix_a_col_start + tile_col1];
            tile_a[tile_row1][tile_col2] = matrix_a[matrix_a_start + matrix_a_row1_start + matrix_a_col_start + tile_col2];
            tile_a[tile_row2][tile_col1] = matrix_a[matrix_a_start + matrix_a_row2_start + matrix_a_col_start + tile_col1];
            tile_a[tile_row2][tile_col2] = matrix_a[matrix_a_start + matrix_a_row2_start + matrix_a_col_start + tile_col2];
        }
        else if (matrix_c_row1 < a_height) {
            const unsigned int matrix_a_start = start * a_width;
            const unsigned int matrix_a_row1_start = matrix_c_row1 * a_width;
            const unsigned int matrix_a_col_start = tile * TILE_SIZE;
            tile_a[tile_row1][tile_col1] = matrix_a[matrix_a_start + matrix_a_row1_start + matrix_a_col_start + tile_col1];
            tile_a[tile_row1][tile_col2] = matrix_a[matrix_a_start + matrix_a_row1_start + matrix_a_col_start + tile_col2];
            tile_a[tile_row2][tile_col1] = 0;
            tile_a[tile_row2][tile_col2] = 0;
        }
        else {
            tile_a[tile_row1][tile_col1] = 0;
            tile_a[tile_row1][tile_col2] = 0;
            tile_a[tile_row2][tile_col1] = 0;
            tile_a[tile_row2][tile_col2] = 0;
        }

        const unsigned int matrix_b_start1 = (tile * TILE_SIZE + tile_row1) * b_width;
        const unsigned int matrix_b_start2 = (tile * TILE_SIZE + tile_row2) * b_width;

        tile_b1[tile_row1][tile_col1] = matrix_b[matrix_b_start1 + matrix_c_col1];
        tile_b1[tile_row1][tile_col2] = matrix_b[matrix_b_start1 + matrix_c_col2];
        tile_b2[tile_row1][tile_col1] = matrix_b[matrix_b_start1 + matrix_c_col3];
        tile_b2[tile_row1][tile_col2] = matrix_b[matrix_b_start1 + matrix_c_col4];
        tile_b3[tile_row1][tile_col1] = matrix_b[matrix_b_start1 + matrix_c_col5];
        tile_b3[tile_row1][tile_col2] = matrix_b[matrix_b_start1 + matrix_c_col6];
        tile_b4[tile_row1][tile_col1] = matrix_b[matrix_b_start1 + matrix_c_col7];
        tile_b4[tile_row1][tile_col2] = matrix_b[matrix_b_start1 + matrix_c_col8];

        tile_b1[tile_row2][tile_col1] = matrix_b[matrix_b_start2 + matrix_c_col1];
        tile_b1[tile_row2][tile_col2] = matrix_b[matrix_b_start2 + matrix_c_col2];
        tile_b2[tile_row2][tile_col1] = matrix_b[matrix_b_start2 + matrix_c_col3];
        tile_b2[tile_row2][tile_col2] = matrix_b[matrix_b_start2 + matrix_c_col4];
        tile_b3[tile_row2][tile_col1] = matrix_b[matrix_b_start2 + matrix_c_col5];
        tile_b3[tile_row2][tile_col2] = matrix_b[matrix_b_start2 + matrix_c_col6];
        tile_b4[tile_row2][tile_col1] = matrix_b[matrix_b_start2 + matrix_c_col7];
        tile_b4[tile_row2][tile_col2] = matrix_b[matrix_b_start2 + matrix_c_col8];

        __syncthreads();
        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            const float a1 = tile_a[tile_row1][k];
            const float a2 = tile_a[tile_row2][k];
            c_01 += a1 * tile_b1[k][tile_col1];
            c_02 += a1 * tile_b1[k][tile_col2];
            c_03 += a1 * tile_b2[k][tile_col1];
            c_04 += a1 * tile_b2[k][tile_col2];
            c_05 += a1 * tile_b3[k][tile_col1];
            c_06 += a1 * tile_b3[k][tile_col2];
            c_07 += a1 * tile_b4[k][tile_col1];
            c_08 += a1 * tile_b4[k][tile_col2];

            c_09 += a2 * tile_b1[k][tile_col1];
            c_10 += a2 * tile_b1[k][tile_col2];
            c_11 += a2 * tile_b2[k][tile_col1];
            c_12 += a2 * tile_b2[k][tile_col2];
            c_13 += a2 * tile_b3[k][tile_col1];
            c_14 += a2 * tile_b3[k][tile_col2];
            c_15 += a2 * tile_b4[k][tile_col1];
            c_16 += a2 * tile_b4[k][tile_col2];
        }
        __syncthreads();
    }

    // relu is included in here
    if (matrix_c_row2 < c_height) {
        const unsigned int matrix_c_start = start * c_width;
        const unsigned int matrix_c_row_start1 = matrix_c_row1 * c_width;
        const unsigned int matrix_c_row_start2 = matrix_c_row2 * c_width;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col1] = (c_01 <= 0) ? 0 : c_01;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col2] = (c_02 <= 0) ? 0 : c_02;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col3] = (c_03 <= 0) ? 0 : c_03;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col4] = (c_04 <= 0) ? 0 : c_04;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col5] = (c_05 <= 0) ? 0 : c_05;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col6] = (c_06 <= 0) ? 0 : c_06;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col7] = (c_07 <= 0) ? 0 : c_07;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col8] = (c_08 <= 0) ? 0 : c_08;

        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col1] = (c_09 <= 0) ? 0 : c_09;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col2] = (c_10 <= 0) ? 0 : c_10;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col3] = (c_11 <= 0) ? 0 : c_11;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col4] = (c_12 <= 0) ? 0 : c_12;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col5] = (c_13 <= 0) ? 0 : c_13;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col6] = (c_14 <= 0) ? 0 : c_14;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col7] = (c_15 <= 0) ? 0 : c_15;
        matrix_c[matrix_c_start + matrix_c_row_start2 + matrix_c_col8] = (c_16 <= 0) ? 0 : c_16;
    }
    else if (matrix_c_row1 < c_height) {
        const unsigned int matrix_c_start = start * c_width;
        const unsigned int matrix_c_row_start1 = matrix_c_row1 * c_width;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col1] = (c_01 <= 0) ? 0 : c_01;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col2] = (c_02 <= 0) ? 0 : c_02;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col3] = (c_03 <= 0) ? 0 : c_03;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col4] = (c_04 <= 0) ? 0 : c_04;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col5] = (c_05 <= 0) ? 0 : c_05;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col6] = (c_06 <= 0) ? 0 : c_06;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col7] = (c_07 <= 0) ? 0 : c_07;
        matrix_c[matrix_c_start + matrix_c_row_start1 + matrix_c_col8] = (c_08 <= 0) ? 0 : c_08;
    }
}

/*
    fully_forward2(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int start)
    DESCRIPTION:
        Kernel function for tile based matrix multiplication specialized for fully forward.
    INPUT:
        matrix_a - pointer to the e array
        matrix_b - pointer to the fc2 array
        matrix_c - pointer to the f array
        a_height - the count of numbers of the current batch
        start - the first number's index of the current batch
*/
__global__ void fully_forward2(const float *matrix_a, const float *matrix_b, float *matrix_c, const unsigned int a_height, const unsigned int start) {

    const unsigned int a_width = FC2_ROWS;
    const unsigned int b_height = a_width;
    const unsigned int b_width = FC2_COLS;
    const unsigned int c_height = a_height;
    const unsigned int c_width = b_width;

    const unsigned int matrix_c_row = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int matrix_c_col = threadIdx.x;

    // each block is 32 * 32
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
    const unsigned int tile_row = threadIdx.y;
    const unsigned int tile_col = threadIdx.x;

    float c = 0;

    // load the tiles and accumulate the product
    for (unsigned int tile = 0; tile < FC2_ROWS / TILE_SIZE; tile++) {
        if ((matrix_c_row < a_height) && (tile * TILE_SIZE + tile_col < a_width)) {
            tile_a[tile_row][tile_col] = matrix_a[matrix_c_row * a_width + tile * TILE_SIZE + tile_col + start * a_width];
        }
        else {
            tile_a[tile_row][tile_col] = 0.0f;
        }
        if ((tile * TILE_SIZE + tile_row < b_height) && (matrix_c_col < b_width)) {
            tile_b[tile_row][tile_col] = matrix_b[(tile * TILE_SIZE + tile_row) * b_width + matrix_c_col];
        }
        else {
            tile_b[tile_row][tile_col] = 0.0f;
        }

        __syncthreads();
        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            c += tile_a[tile_row][k] * tile_b[k][tile_col];
        }
        __syncthreads();
    }

    // don't need relu here
    if ((matrix_c_row < c_height) && (matrix_c_col < c_width)) {
        matrix_c[matrix_c_row * c_width + matrix_c_col + start * c_width] = c;
    }
}

/*
    arg_max(const float *input, unsigned int *output, const unsigned int input_len, const unsigned int start)
    DESCRIPTION:
        Kernel function for finding the result.
    INPUT:
        input - pointer to the f array
        output - pointer to the output array
        input_len - length of the f array
        start - the first number's index of the current batch
*/
__global__ void arg_max(const float *input, unsigned int *output, const unsigned int input_len, const unsigned int start) {
    const unsigned int t = (start + blockIdx.x * blockDim.x + threadIdx.x) * NUM_DIGITS;
    if (t < input_len) {
        unsigned int max_idx = 0;
        float max = input[t];
        for (unsigned int i = 1; i < NUM_DIGITS; i++) {
            const float temp = input[t + i];
            if (temp > max) {
                max = temp;
                max_idx = i;
            }
        }
        output[t / NUM_DIGITS] = max_idx;
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

    // this chunck of code is kind of extra but necessary
    // since the provided dataset has some very weird dimensions
    // we transform them to more human understandable ones
    dim3 block_dim_conv(CONV_COLS, CONV_ROWS, CONV1_OUTPUT_CHANNELS);
    dim3 block_dim_fc1(d_dims[3], d_dims[2], d_dims[1]);
    cudaMemcpyAsync(conv1_device_, conv1, conv1_len * sizeof(float), cudaMemcpyHostToDevice, streams[STREAM_IDX_CONV1]);
    transform_conv1<<<CONV1_INPUT_CHANNELS, block_dim_conv, 0, streams[STREAM_IDX_CONV1]>>>(conv1_device_, conv1_device);
    cudaMemcpyAsync(conv2_device_, conv2, conv2_len * sizeof(float), cudaMemcpyHostToDevice, streams[STREAM_IDX_CONV2]);
    transform_conv2<<<CONV2_OUTPUT_CHANNELS, block_dim_conv, 0, streams[STREAM_IDX_CONV2]>>>(conv2_device_, conv2_device);
    cudaMemcpyAsync(fc1_device_, fc1, fc1_len * sizeof(float), cudaMemcpyHostToDevice, streams[STREAM_IDX_FC1]);
    transform_fc1<<<FC1_COLS, block_dim_fc1, 0, streams[STREAM_IDX_FC1]>>>(fc1_device_, fc1_device);
    cudaMemcpyAsync(fc2_device, fc2, fc2_len * sizeof(float), cudaMemcpyHostToDevice, streams[STREAM_IDX_FC2]);

    // s is the index for streams, reset to 0 for every kernel lunch such that all kernels will be streamlined
    // i.e. a new kernel that is using stream x will only use the data calculated by the previous kernel that is also using stream x
    unsigned int s = 0;

    // batch num is how much data we will be using for each kernel, need to change only once for unroll2
    unsigned int batch_num = BATCH_NUM_PER_STREAM;

    // unroll kernel 1, this will unroll from input dataset, also use async memcpy to make it more efficient
    for (unsigned int start = 0; start < input_dims[0]; start += batch_num) {
        const unsigned int grid_size = min(batch_num, input_dims[0] - start);
        const unsigned int offset = start * INPUT_NUM_ELEMENTS;
        cudaMemcpyAsync(input_device + offset, input + offset, grid_size * INPUT_NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        unroll1<<<grid_size, BLOCK_SIZE, 0, streams[s]>>>(input_device, input_unroll_device, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // matrix multiplication kernel 1, you know, multiply two matrices
    s = 0;
    dim3 block_dim_mm(HALF_TILE_SIZE, HALF_TILE_SIZE, 1);
    cudaStreamSynchronize(streams[STREAM_IDX_CONV1]);
    for (unsigned int start = 0; start < input_unroll_dims[0]; start += batch_num) {
        dim3 grid_dim_matrix_mul1(MATRIX_MUL1_BLOCKS_PER_NUM, min(batch_num, input_unroll_dims[0] - start), 1);
        matrix_multiplication1<<<grid_dim_matrix_mul1, block_dim_mm, 0, streams[s]>>>(conv1_device, input_unroll_device, a_device, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // average pool kernel 1, err, like the name suggests, it averages stuff...
    s = 0;
    unsigned int layers = MAX_THREADS_PER_BLOCK / B_NUM_ELEMENTS;
    dim3 block_dim_avg1(b_dims[2], b_dims[3], layers);
    for (unsigned int start = 0; start < a_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, a_dims[0] - start);
        const unsigned int grid_size = static_cast<unsigned int>(ceil(todo_count / (float)layers));
        average_pool1<<<grid_size, block_dim_avg1, 0, streams[s]>>>(a_device, b_device, start, start + todo_count);
        s = (s + 1) % NUM_STREAMS;
    }

    // unroll kernel 2, now we have 32 channels per number, so multiply the batch number by 32
    // and we don't need to copy data from anywhere
    s = 0;
    batch_num = batch_num * BATCH_NUM_FACTOR;
    layers = UNROLL2_LAYERS;
    const unsigned int total_channels = b_dims[0] * b_dims[1];
    dim3 block_dim_unroll2(b_unroll_dims[3], layers, 1);
    for (unsigned int start = 0; start < total_channels; start += batch_num) {
        unroll2<<<min(static_cast<unsigned int>(ceil(batch_num / (float)layers)), static_cast<unsigned int>(ceil(total_channels - start) / (float)layers)), block_dim_unroll2, 0, streams[s]>>>(b_device, b_unroll_device, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // matrix multiplication kernel 2, again multiply two matrices
    s = 0;
    batch_num = batch_num / BATCH_NUM_FACTOR;
    cudaStreamSynchronize(streams[STREAM_IDX_CONV2]);
    for (unsigned int start = 0; start < b_unroll_dims[0]; start += batch_num) {
        matrix_multiplication2<<<min(static_cast<unsigned int>(ceil(batch_num / 2.0f)), static_cast<unsigned int>(ceil((b_unroll_dims[0] - start) / 2.0f))), block_dim_mm, 0, streams[s]>>>(conv2_device, b_unroll_device, c_device, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // average pool kernel 2, really, you need to read this to understand average?
    s = 0;
    layers = BLOCK_SIZE / D_NUM_ELEMENTS;
    dim3 block_dim_avg2(d_dims[2], d_dims[3], layers);
    for (unsigned int start = 0; start < c_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, c_dims[0] - start);
        const unsigned int grid_size = static_cast<unsigned int>(ceil(todo_count / (float)layers));
        average_pool2<<<grid_size, block_dim_avg2, 0, streams[s]>>>(c_device, d_device, start, start + todo_count);
        s = (s + 1) % NUM_STREAMS;
    }

    // fully forward kernel 1, even though it's named different, but deep down it's still just matrix multiplication...
    s = 0;
    cudaStreamSynchronize(streams[STREAM_IDX_FC1]);
    for (unsigned int start = 0; start < d_dims2[0]; start += batch_num) {
        const unsigned int a_height = min(batch_num, d_dims2[0] - start);
        fully_forward1<<<static_cast<unsigned int>(ceil(a_height / (float)TILE_SIZE)), block_dim_mm, 0, streams[s]>>>(d_device, fc1_device, e_device, a_height, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // fully forward kernel 2, ok, so this is a different one, jk it's not
    // well, only if you think not including relu is different, then it is different
    s = 0;
    dim3 block_dim_ff(TILE_SIZE, TILE_SIZE, 1);
    cudaStreamSynchronize(streams[STREAM_IDX_FC2]);
    for (unsigned int start = 0; start < e_dims[0]; start += batch_num) {
        const unsigned int a_height = min(batch_num, e_dims[0] - start);
        fully_forward2<<<static_cast<unsigned int>(ceil(a_height / (float)TILE_SIZE)), block_dim_ff, 0, streams[s]>>>(e_device, fc2_device, f_device, a_height, start);
        s = (s + 1) % NUM_STREAMS;
    }

    // arg max kernel, search for the largest number...'s index in the array of arrays, the index is the index of the sub array (between 0 - 9)
    s = 0;
    for (unsigned int start = 0; start < f_dims[0]; start += batch_num) {
        const unsigned int todo_count = min(batch_num, f_dims[0] - start);
        arg_max<<<static_cast<unsigned int>(ceil(todo_count / (float)BLOCK_SIZE)), BLOCK_SIZE, 0, streams[s]>>>(f_device, output_device, f_len, start);
        cudaMemcpyAsync(output + start, output_device + start, todo_count * sizeof(int), cudaMemcpyDeviceToHost, streams[s]);
        s = (s + 1) % NUM_STREAMS;
    }

    // if sync is outside of this function, then boom!
    cudaDeviceSynchronize();
}

/*
    main(unsigned int argc, char **argv)
    DESCRIPTION:
        The main function... lol
    INPUT:
        argc - number of args
        argv - pointer to args
*/
int main(unsigned int argc, char **argv) {

    if (argc != 4) {
        std::cerr << "\nThis program performs the forward opertion step for Convolutional Neural Network(CNN). Sample usage: \n"
            << argv[0] << " test10.hdf5 model.hdf5 10\n";
        return -1;
    }

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

    // init all the streams
    for (unsigned int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // malloc everything...
    cudaMalloc((void **)&all_memory_device, ((conv1_len << 1) + (conv2_len << 1) + (fc1_len << 1) + fc2_len + input_len + input_unroll_len + a_len + b_len + b_unroll_len + c_len + d_len + e_len + f_len) * sizeof(float) + output_len * sizeof(int));
    conv1_device_ = &all_memory_device[0];
    conv2_device_ = &conv1_device_[conv1_len];
    fc1_device_ = &conv2_device_[conv2_len];
    conv1_device = &fc1_device_[fc1_len];
    conv2_device = &conv1_device[conv1_len];
    fc1_device = &conv2_device[conv2_len];
    fc2_device = &fc1_device[fc1_len];
    input_device = &fc2_device[fc2_len];
    input_unroll_device = &input_device[input_len];
    a_device = &input_unroll_device[input_unroll_len];
    b_device = &a_device[a_len];
    b_unroll_device = &b_device[b_len];
    c_device = &b_unroll_device[b_unroll_len];
    d_device = &c_device[c_len];
    e_device = &d_device[d_len];
    f_device = &e_device[e_len];
    output_device = (unsigned int *)&f_device[f_len];

    float *all_memory_host, *input, *conv1, *conv2, *fc1, *fc2;
    unsigned int *output;

    // malloc space for everything in host
    cudaMallocHost((void **)&all_memory_host, (input_len + conv1_len + conv2_len + fc1_len + fc2_len) * sizeof(float) + output_len * sizeof(int));
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

    // don't need to clean up since we stop running everything.... so good
    return 0;
}
