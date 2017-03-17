#!/bin/bash

nvprof --cpu-profiling on --export-profile timeline.nvprof -- ../build/./cuda_cnn ../data/testfull.hdf5 ../data/model.hdf5 10000

nvprof --cpu-profiling on --export-profile analysis.nvprof --analysis-metrics -- ../build/./cuda_cnn ../data/testfull.hdf5 ../data/model.hdf5 10000
