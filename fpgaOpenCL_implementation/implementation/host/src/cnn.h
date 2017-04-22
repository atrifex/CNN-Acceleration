#ifndef __CNN_H__
#define __CNN_H__

#include <iostream>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <sys/time.h>
#include <valarray>
#include <vector>

#include <map>
#include <iostream>
#include <string>

using std::string;
using std::map;
using std::vector;
using std::min;

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "hdf5/include/hdf5.h"

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "range.hpp"
#include "utils.hpp"

#define HALF_TILE_SIZE              16
#define TILE_SIZE                   32
#define DOUBLE_TILE_SIZE            64
#define BLOCK_SIZE                  512
#define POOL_SIZE                   2
#define AVG_COUNT                   (POOL_SIZE * POOL_SIZE)

#define NUM_DIGITS                  10
#define NUM_CMD_QUEUES              16
#define QUEUE_IDX_CONV1             (NUM_CMD_QUEUES - 1)
#define QUEUE_IDX_CONV2             (NUM_CMD_QUEUES - 2)
#define QUEUE_IDX_FC1               (NUM_CMD_QUEUES - 3)
#define QUEUE_IDX_FC2               (NUM_CMD_QUEUES - 4)
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

#define BATCH_NUM_PER_STREAM        10000
#define BATCH_NUM_FACTOR            32
#define CONSTANT_MEM_SIZE           (64 * 1024)

#define MATRIX_MUL1_BLOCKS_PER_NUM  6
#define MATRIX_MUL1_COLS_PER_BLOCK  (A_NUM_ELEMENTS / MATRIX_MUL1_BLOCKS_PER_NUM)
#define MATRIX_MUL1_SKIP1           (CONV_NUM_ELEMENTS * A_NUM_ELEMENTS)
#define MATRIX_MUL1_SKIP2           (CONV1_OUTPUT_CHANNELS * A_NUM_ELEMENTS)
#define MATRIX_MUL2_NUMS_PER_BLOCK  2
#define UNROLL2_LAYERS              (BLOCK_SIZE / C_NUM_ELEMENTS)
#define FULLY_FORWARD1_TILE_NUM     (FC1_ROWS / TILE_SIZE)

#endif
