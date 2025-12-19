#ifndef __TILED_CONV_HPP__
#define __TILED_CONV_HPP__

#include <ap_fixed.h>
#include <hls_math.h>

//--------------------------------------------------------------------------
// Fixed-Point Data Types
//--------------------------------------------------------------------------
typedef ap_fixed<16, 12> fm_t, wt_t;
typedef ap_fixed<32, 24> acc_t;

//--------------------------------------------------------------------------
// ARCHITECTURE PARAMETERS
//--------------------------------------------------------------------------
#define K 3 // Kernel Size

// FC Layer Parameters (Based on 256 channels * 2x2 spatial)
#define FC_IN_SIZE 1024  // Flattened output of Pool 4 (256 * 2 * 2)
#define FC1_OUT_SIZE 256 // Output of FC layer 1
#define FC_OUT_SIZE 10   // Final output classes

// Tiling constants
#define MAX_TILE_HW 8   
#define MAX_MARGIN 1   
#define MAX_IN_BUF_HW (MAX_TILE_HW + 2 * MAX_MARGIN)
#define MAX_OUT_BUF_DEPTH 8
#define MAX_M 256  // Maximum input channels

// BatchNorm constants
#define BN_EPSILON 1e-5f

// Feature map buffer sizes for ping-pong buffering
#define FM_BUF_SIZE 32768  // 32x32x32 max

// Helper macros
#define GET_3D_ADDR(m, r, c, width) ((m) * (width) * (width) + (r) * (width) + (c))

//==========================================================================
// Function Prototypes
//==========================================================================

// Top-Level Inference Function
void reduced_vgg_inference(
	fm_t *input_fm,
	// Conv & BN L1-1 (W1)
	wt_t *W_1, wt_t *B_1, wt_t *G_1, wt_t *T_1, wt_t *M_1, wt_t *V_1,
	// Conv & BN L1-2 (W2)
	wt_t *W_2, wt_t *B_2, wt_t *G_2, wt_t *T_2, wt_t *M_2, wt_t *V_2,
	// Conv & BN L2-1 (W3)
	wt_t *W_3, wt_t *B_3, wt_t *G_3, wt_t *T_3, wt_t *M_3, wt_t *V_3,
	// Conv & BN L2-2 (W4)
	wt_t *W_4, wt_t *B_4, wt_t *G_4, wt_t *T_4, wt_t *M_4, wt_t *V_4,
	// Conv & BN L3-1 (W5)
	wt_t *W_5, wt_t *B_5, wt_t *G_5, wt_t *T_5, wt_t *M_5, wt_t *V_5,
	// Conv & BN L3-2 (W6)
	wt_t *W_6, wt_t *B_6, wt_t *G_6, wt_t *T_6, wt_t *M_6, wt_t *V_6,
	// Conv & BN L4-1 (W7)
	wt_t *W_7, wt_t *B_7, wt_t *G_7, wt_t *T_7, wt_t *M_7, wt_t *V_7,
	// Conv & BN L4-2 (W8)
	wt_t *W_8, wt_t *B_8, wt_t *G_8, wt_t *T_8, wt_t *M_8, wt_t *V_8,
	// FC Layers
	wt_t *W_FC1, wt_t *B_FC1, wt_t *W_FC2, wt_t *B_FC2,
	fm_t *output_fm);

// Utility Function Prototypes
void tiled_conv_layer(
	fm_t *input_fm,
	wt_t *weights,
	wt_t *biases,
	fm_t *output_fm,
	int M, int I, int N, int O, int S, int P,
	int TILE_HW, int OUT_BUF_DEPTH);

void batch_norm_relu_layer(
	fm_t *input_fm,
	wt_t *gamma, wt_t *beta, wt_t *mean, wt_t *variance,
	fm_t *output_fm,
	int M, int I);

void max_pool_2x2(
	fm_t *input_fm,
	fm_t *output_fm,
	int M, int I, int O);

void fully_connected_layer(
	fm_t *input_fm,
	wt_t *weights,
	wt_t *biases,
	fm_t *output_fm,
	int IN_SIZE, int OUT_SIZE);

#endif

