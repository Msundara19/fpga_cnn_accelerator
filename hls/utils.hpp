/**
 * @file utils.hpp
 * @brief Utility function declarations for Reduced VGG - OPTIMIZED VERSION
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "tiled_conv.hpp"

//==============================================================================
// CONVOLUTION UTILITIES
//==============================================================================

void load_input_tile_from_DRAM(
	fm_t *in_fm,
	fm_t in_fm_buf[MAX_M][MAX_IN_BUF_HW][MAX_IN_BUF_HW],
	int tr, int tc,
	int M, int I, int P, int TILE_HW, int MARGIN);

void load_layer_params_from_DRAM(
	wt_t *weights,
	wt_t *biases,
	wt_t weights_buf[MAX_OUT_BUF_DEPTH][MAX_M][K][K],
	wt_t biases_buf[MAX_OUT_BUF_DEPTH],
	int kernel_group,
	int M, int N, int OUT_BUF_DEPTH);

void conv_3x3(
	fm_t X_buf[MAX_M][MAX_IN_BUF_HW][MAX_IN_BUF_HW],
	wt_t W_buf[MAX_OUT_BUF_DEPTH][MAX_M][K][K],
	wt_t B_buf[MAX_OUT_BUF_DEPTH],
	fm_t Y_buf[MAX_OUT_BUF_DEPTH][MAX_TILE_HW][MAX_TILE_HW],
	int M, int TILE_HW, int OUT_BUF_DEPTH);

void store_output_tile_to_DRAM(
	fm_t Y_buf[MAX_OUT_BUF_DEPTH][MAX_TILE_HW][MAX_TILE_HW],
	fm_t *out_fm,
	int tr, int tc, int kernel_group,
	int N, int O, int TILE_HW, int OUT_BUF_DEPTH);

//==============================================================================
// POOLING UTILITIES
//==============================================================================

void max_pool_2x2(
	fm_t *input_fm,
	fm_t *output_fm,
	int M, int I, int O);

//==============================================================================
// FULLY CONNECTED UTILITIES
//==============================================================================

void fully_connected_layer(
	fm_t *input_fm,
	wt_t *weights,
	wt_t *biases,
	fm_t *output_fm,
	int IN_SIZE, int OUT_SIZE);

#endif


