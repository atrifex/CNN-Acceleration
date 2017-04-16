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

/*
    transform_conv1(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv1 to the normal form.
    INPUT:
        input - pointer to the old conv1 array
        output - pointer to the new conv1 array
*/
__kernel void transform_conv1(__global float *input, __global float *output) {

    // old dimmensions are (row=5, col=5, input_channel=1, output_channel=32)
    // new dimmensions are (output_channel=32, input_channel=1, row=5, col=5)
    output[get_local_id(2) * CONV_ROWS * CONV_COLS + get_local_id(1) * CONV_COLS + get_local_id(0)] =
        input[get_local_id(1) * CONV_COLS * CONV1_OUTPUT_CHANNELS + get_local_id(0) * CONV1_OUTPUT_CHANNELS + get_local_id(2)];
}

/*
    transform_conv2(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv2 to the normal form.
    INPUT:
        input - pointer to the old conv2 array
        output - pointer to the new conv2 array
*/
__kernel void transform_conv2(__global float *input, __global float *output) {

    // old dimmensions are (row=5, col=5, input_channel=32, output_channel=64)
    // new dimmensions are (output_channel=64, input_channel=32, row=5, col=5)
    output[get_group_id(0) * CONV2_INPUT_CHANNELS * CONV_ROWS * CONV_COLS +
            get_local_id(2) * CONV_ROWS * CONV_COLS + get_local_id(1) * CONV_COLS + get_local_id(0)] =
        input[get_local_id(1) * CONV_COLS * CONV2_INPUT_CHANNELS * CONV2_OUTPUT_CHANNELS +
            get_local_id(0) * CONV2_INPUT_CHANNELS * CONV2_OUTPUT_CHANNELS + get_local_id(2) * CONV2_OUTPUT_CHANNELS + get_group_id(0)];
}

/*
    transform_fc1(const float *input, float *output)
    DESCRIPTION:
        Kernel function for transforming the dimmensions of conv2 to the normal form.
    INPUT:
        input - pointer to the old fc1 array
        output - pointer to the new fc1 array
*/
__kernel void transform_fc1(__global float *input, __global float *output) {

    // old dimmensions are (row=1024, col=128), the sub dimmensions for row are (sub_row=4, sub_col=4, channel=64)
    // old dimmensions are (row=1024, col=128), the sub dimmensions for row are (channel=64, sub_row=4, sub_col=4)
    output[get_local_id(2) * D_ROWS * D_COLS * FC1_COLS + get_local_id(1) * D_COLS * FC1_COLS +
            get_local_id(0) * FC1_COLS + get_group_id(0)] =
        input[get_local_id(1) * D_COLS * CONV2_OUTPUT_CHANNELS * FC1_COLS +
            get_local_id(0) * CONV2_OUTPUT_CHANNELS * FC1_COLS + get_local_id(2) * FC1_COLS + get_group_id(0)];
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
__kernel void unroll1(__global float *x, __global float *x_unroll, const unsigned int start) {

    // 5 * 5 from filter size (per channel)
    const unsigned int x_unroll_height = CONV_NUM_ELEMENTS;

    // 24 * 24 from new size after multiplication
    const unsigned int x_unroll_width = A_NUM_ELEMENTS;

    // every block unrolls 1 channel
    // channel_idx is the universal index of channels for all input data
    const unsigned int channel_idx = start + get_group_id(0);

    // the amount of addresses in one channel
    const unsigned int address_per_channel = x_unroll_height * x_unroll_width;

    // load a channel of inputs into shared memory
    __local float x_shared[INPUT_NUM_ELEMENTS];
    const unsigned int x_base = channel_idx * INPUT_NUM_ELEMENTS;
    for (unsigned int i = get_local_id(0); i < INPUT_NUM_ELEMENTS; i += get_local_size(0)) {
        x_shared[i] = x[x_base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int t = get_local_id(0);
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
__kernel void unroll2(__global float *x, __global float *x_unroll, const unsigned int start) {

    // 5 * 5 from filter size (per channel)
    const unsigned int x_unroll_height = CONV_NUM_ELEMENTS;

    // 8 * 8 from new size after multiplication
    const unsigned int x_unroll_width = C_NUM_ELEMENTS;

    // every block unrolls 8 channelS
    // channel_idx is the universal index of channels for all input data
    const unsigned int channel_idx = start + get_group_id(0) * get_local_size(1) + get_local_id(1);

    // the amount of addresses in one channel
    const unsigned int address_per_channel = x_unroll_height * x_unroll_width;

    // load a channel of inputs into shared memory
    __local float x_shared[UNROLL2_LAYERS][B_NUM_ELEMENTS];
    const unsigned int x_base = channel_idx * B_NUM_ELEMENTS;
    for (unsigned int i = get_local_id(0); i < B_NUM_ELEMENTS; i += get_local_size(0)) {
        x_shared[get_local_id(1)][i] = x[x_base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int t = get_local_id(0);

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
            x_unroll[x_unroll_offset] = x_shared[get_local_id(1)][x_start + (x_col_base + j)];
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
__kernel void matrix_multiplication1(__global float *matrix_a, __global float *matrix_b, __global float *matrix_c, const unsigned int start) {

    // matrix_a is not needed here since it's in constant memory
    const unsigned int a_width = CONV1_NUM_ELEMENTS_IN;
    const unsigned int b_height = CONV_NUM_ELEMENTS;
    const unsigned int b_width = A_NUM_ELEMENTS;
    const unsigned int c_height = CONV1_OUTPUT_CHANNELS;
    const unsigned int c_width = b_width;

    // each block deals with 1/6 of a number
    // block index in x direction is for one of these 6 blocks
    // block index in y direction is for number indices
    const unsigned int num_idx = get_group_id(1);

    const unsigned int matrix_c_row1 = get_local_id(1);
    const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;

    const unsigned int matrix_c_col1 = get_group_id(0) * MATRIX_MUL1_COLS_PER_BLOCK + get_local_id(0);
    const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col5 = matrix_c_col4 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col6 = matrix_c_col5 + HALF_TILE_SIZE;

    // size of unrolled input per channel is 25 * 576, each block is 16 * 16
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b_1[TILE_SIZE][TILE_SIZE];
    __local float tile_b_2[TILE_SIZE][TILE_SIZE];
    __local float tile_b_3[TILE_SIZE][TILE_SIZE];

    const unsigned int tile_row1 = get_local_id(1);
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_col1 = get_local_id(0);
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

    barrier(CLK_LOCAL_MEM_FENCE);
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
__kernel void matrix_multiplication2(__global float *matrix_a, __global float *matrix_b, __global float *matrix_c, const unsigned int start) {

    const unsigned int a_width = CONV2_NUM_ELEMENTS_IN;
    const unsigned int b_height = a_width;
    const unsigned int b_width = C_NUM_ELEMENTS;
    const unsigned int c_height = CONV2_OUTPUT_CHANNELS;
    const unsigned int c_width = b_width;

    // const unsigned int matrix_c_row1 = get_local_id(1);
    // const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_row3 = matrix_c_row2 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_row4 = matrix_c_row3 + HALF_TILE_SIZE;

    // const unsigned int matrix_c_col1 = get_local_id(0);
    // const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    // const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;

    // every block deals with 2 numbers
    const unsigned int num_idx = get_group_id(0) * MATRIX_MUL2_NUMS_PER_BLOCK;

    // each block is 16 * 16,
    // we load a tile of 64 * 32 from matrix a, since matrix a is 64 * 800
    // we also load a tile of 32 * 64 from matrix b, since matrix b is 800 * 64 per number
    __local float tile_a[DOUBLE_TILE_SIZE][TILE_SIZE];
    __local float tile_b1[TILE_SIZE][DOUBLE_TILE_SIZE];
    __local float tile_b2[TILE_SIZE][DOUBLE_TILE_SIZE];

    const unsigned int tile_row1 = get_local_id(1);
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_row3 = tile_row2 + HALF_TILE_SIZE;
    const unsigned int tile_row4 = tile_row3 + HALF_TILE_SIZE;

    const unsigned int tile_col1 = get_local_id(0);
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

        barrier(CLK_LOCAL_MEM_FENCE);
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
        barrier(CLK_LOCAL_MEM_FENCE);
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
__kernel void average_pool1(__global float *x, __global float *y, const unsigned int start, const unsigned int end) {

    const float count = AVG_COUNT;
    const unsigned int row = get_local_id(1);
    const unsigned int col = get_local_id(0);

    // the index of the number this thread is averaging
    const unsigned int num_idx = start + get_group_id(0) * get_local_size(2) + get_local_id(2);

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
__kernel void average_pool2(__global float *x, __global float *y, const unsigned int start, const unsigned int end) {

    const float count = AVG_COUNT;
    const unsigned int row = get_local_id(1);
    const unsigned int col = get_local_id(0);

    // the index of the number this thread is averaging
    const unsigned int num_idx = start + get_group_id(0) * get_local_size(2) + get_local_id(2);

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
__kernel void fully_forward1(__global float *matrix_a, __global float *matrix_b, __global float *matrix_c, const unsigned int a_height, const unsigned int start) {

    const unsigned int a_width = FC1_ROWS;
    const unsigned int b_width = FC1_COLS;
    const unsigned int c_height = a_height;
    const unsigned int c_width = b_width;

    const unsigned int matrix_c_row1 = get_group_id(0) * TILE_SIZE + get_local_id(1);
    const unsigned int matrix_c_row2 = matrix_c_row1 + HALF_TILE_SIZE;

    const unsigned int matrix_c_col1 = get_local_id(0);
    const unsigned int matrix_c_col2 = matrix_c_col1 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col3 = matrix_c_col2 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col4 = matrix_c_col3 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col5 = matrix_c_col4 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col6 = matrix_c_col5 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col7 = matrix_c_col6 + HALF_TILE_SIZE;
    const unsigned int matrix_c_col8 = matrix_c_col7 + HALF_TILE_SIZE;

    // each block is 16 * 16
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b1[TILE_SIZE][TILE_SIZE];
    __local float tile_b2[TILE_SIZE][TILE_SIZE];
    __local float tile_b3[TILE_SIZE][TILE_SIZE];
    __local float tile_b4[TILE_SIZE][TILE_SIZE];

    const unsigned int tile_row1 = get_local_id(1);
    const unsigned int tile_row2 = tile_row1 + HALF_TILE_SIZE;
    const unsigned int tile_col1 = get_local_id(0);
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

        barrier(CLK_LOCAL_MEM_FENCE);
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
        barrier(CLK_LOCAL_MEM_FENCE);
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
__kernel void fully_forward2(__global float *matrix_a, __global float *matrix_b, __global float *matrix_c, const unsigned int a_height, const unsigned int start) {

    const unsigned int a_width = FC2_ROWS;
    const unsigned int b_height = a_width;
    const unsigned int b_width = FC2_COLS;
    const unsigned int c_height = a_height;
    const unsigned int c_width = b_width;

    const unsigned int matrix_c_row = get_group_id(0) * get_local_size(0) + get_local_id(1);
    const unsigned int matrix_c_col = get_local_id(0);

    // each block is 32 * 32
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];
    const unsigned int tile_row = get_local_id(1);
    const unsigned int tile_col = get_local_id(0);

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

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            c += tile_a[tile_row][k] * tile_b[k][tile_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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
__kernel void arg_max(__global float *input, __global unsigned int *output, const unsigned int input_len, const unsigned int start) {
    const unsigned int t = (start + get_group_id(0) * get_local_size(0) + get_local_id(0)) * NUM_DIGITS;
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
