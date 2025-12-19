/**
 * @file tiled_conv.cpp - RESOURCE-CONSTRAINED VERSION
 * @brief Optimized for Zynq-7020 limits (< 80% utilization)
 */

#include "tiled_conv.hpp"

void reduced_vgg_inference(
    fm_t *input_fm,
    wt_t *W_1, wt_t *B_1, wt_t *G_1, wt_t *T_1, wt_t *M_1, wt_t *V_1,
    wt_t *W_2, wt_t *B_2, wt_t *G_2, wt_t *T_2, wt_t *M_2, wt_t *V_2,
    wt_t *W_3, wt_t *B_3, wt_t *G_3, wt_t *T_3, wt_t *M_3, wt_t *V_3,
    wt_t *W_4, wt_t *B_4, wt_t *G_4, wt_t *T_4, wt_t *M_4, wt_t *V_4,
    wt_t *W_5, wt_t *B_5, wt_t *G_5, wt_t *T_5, wt_t *M_5, wt_t *V_5,
    wt_t *W_6, wt_t *B_6, wt_t *G_6, wt_t *T_6, wt_t *M_6, wt_t *V_6,
    wt_t *W_7, wt_t *B_7, wt_t *G_7, wt_t *T_7, wt_t *M_7, wt_t *V_7,
    wt_t *W_8, wt_t *B_8, wt_t *G_8, wt_t *T_8, wt_t *M_8, wt_t *V_8,
    wt_t *W_FC1, wt_t *B_FC1, wt_t *W_FC2, wt_t *B_FC2,
    fm_t *output_fm)
{
    // ============================================
    // INTERFACE PRAGMAS
    // ============================================
    #pragma HLS INTERFACE m_axi port=input_fm offset=slave bundle=gmem depth=3072
    #pragma HLS INTERFACE m_axi port=output_fm offset=slave bundle=gmem depth=10
    
    #pragma HLS INTERFACE m_axi port=W_1 offset=slave bundle=gmem depth=864
    #pragma HLS INTERFACE m_axi port=B_1 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=G_1 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=T_1 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=M_1 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=V_1 offset=slave bundle=gmem depth=32
    
    #pragma HLS INTERFACE m_axi port=W_2 offset=slave bundle=gmem depth=9216
    #pragma HLS INTERFACE m_axi port=B_2 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=G_2 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=T_2 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=M_2 offset=slave bundle=gmem depth=32
    #pragma HLS INTERFACE m_axi port=V_2 offset=slave bundle=gmem depth=32
    
    #pragma HLS INTERFACE m_axi port=W_3 offset=slave bundle=gmem depth=18432
    #pragma HLS INTERFACE m_axi port=B_3 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=G_3 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=T_3 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=M_3 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=V_3 offset=slave bundle=gmem depth=64
    
    #pragma HLS INTERFACE m_axi port=W_4 offset=slave bundle=gmem depth=36864
    #pragma HLS INTERFACE m_axi port=B_4 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=G_4 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=T_4 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=M_4 offset=slave bundle=gmem depth=64
    #pragma HLS INTERFACE m_axi port=V_4 offset=slave bundle=gmem depth=64
    
    #pragma HLS INTERFACE m_axi port=W_5 offset=slave bundle=gmem depth=73728
    #pragma HLS INTERFACE m_axi port=B_5 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=G_5 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=T_5 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=M_5 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=V_5 offset=slave bundle=gmem depth=128
    
    #pragma HLS INTERFACE m_axi port=W_6 offset=slave bundle=gmem depth=147456
    #pragma HLS INTERFACE m_axi port=B_6 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=G_6 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=T_6 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=M_6 offset=slave bundle=gmem depth=128
    #pragma HLS INTERFACE m_axi port=V_6 offset=slave bundle=gmem depth=128
    
    #pragma HLS INTERFACE m_axi port=W_7 offset=slave bundle=gmem depth=294912
    #pragma HLS INTERFACE m_axi port=B_7 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=G_7 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=T_7 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=M_7 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=V_7 offset=slave bundle=gmem depth=256
    
    #pragma HLS INTERFACE m_axi port=W_8 offset=slave bundle=gmem depth=589824
    #pragma HLS INTERFACE m_axi port=B_8 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=G_8 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=T_8 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=M_8 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=V_8 offset=slave bundle=gmem depth=256
    
    #pragma HLS INTERFACE m_axi port=W_FC1 offset=slave bundle=gmem depth=262144
    #pragma HLS INTERFACE m_axi port=B_FC1 offset=slave bundle=gmem depth=256
    #pragma HLS INTERFACE m_axi port=W_FC2 offset=slave bundle=gmem depth=2560
    #pragma HLS INTERFACE m_axi port=B_FC2 offset=slave bundle=gmem depth=10
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // ============================================
    // INTERNAL BUFFERS - REDUCED PARTITIONING
    // ============================================
    fm_t fm_buf1[FM_BUF_SIZE];
    fm_t fm_buf2[FM_BUF_SIZE];
    fm_t fm_fc1_out[FC1_OUT_SIZE];
    
    // Reduced from factor=16 to factor=4 (4x less resources)
    #pragma HLS ARRAY_PARTITION variable=fm_buf1 cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=fm_buf2 cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=fm_fc1_out cyclic factor=8 dim=1
    
    #pragma HLS BIND_STORAGE variable=fm_buf1 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=fm_buf2 type=ram_2p impl=bram

    // ============================================
    // VGG INFERENCE PIPELINE
    // ============================================
    
    // Block 0: 3  32 channels @ 32x32
    tiled_conv_layer(input_fm, W_1, B_1, fm_buf1, 3, 32, 32, 32, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf1, G_1, T_1, M_1, V_1, fm_buf2, 32, 32);
    
    tiled_conv_layer(fm_buf2, W_2, B_2, fm_buf1, 32, 32, 32, 32, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf1, G_2, T_2, M_2, V_2, fm_buf2, 32, 32);
    
    max_pool_2x2(fm_buf2, fm_buf1, 32, 32, 16);
    
    // Block 1: 32  64 channels @ 16x16
    tiled_conv_layer(fm_buf1, W_3, B_3, fm_buf2, 32, 16, 64, 16, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf2, G_3, T_3, M_3, V_3, fm_buf1, 64, 16);
    
    tiled_conv_layer(fm_buf1, W_4, B_4, fm_buf2, 64, 16, 64, 16, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf2, G_4, T_4, M_4, V_4, fm_buf1, 64, 16);
    
    max_pool_2x2(fm_buf1, fm_buf2, 64, 16, 8);
    
    // Block 2: 64  128 channels @ 8x8
    tiled_conv_layer(fm_buf2, W_5, B_5, fm_buf1, 64, 8, 128, 8, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf1, G_5, T_5, M_5, V_5, fm_buf2, 128, 8);
    
    tiled_conv_layer(fm_buf2, W_6, B_6, fm_buf1, 128, 8, 128, 8, 1, 1, 8, 8);
    batch_norm_relu_layer(fm_buf1, G_6, T_6, M_6, V_6, fm_buf2, 128, 8);
    
    max_pool_2x2(fm_buf2, fm_buf1, 128, 8, 4);
    
    // Block 3: 128  256 channels @ 4x4
    tiled_conv_layer(fm_buf1, W_7, B_7, fm_buf2, 128, 4, 256, 4, 1, 1, 4, 8);
    batch_norm_relu_layer(fm_buf2, G_7, T_7, M_7, V_7, fm_buf1, 256, 4);
    
    tiled_conv_layer(fm_buf1, W_8, B_8, fm_buf2, 256, 4, 256, 4, 1, 1, 4, 8);
    batch_norm_relu_layer(fm_buf2, G_8, T_8, M_8, V_8, fm_buf1, 256, 4);
    
    // Fully Connected Layers
    fully_connected_layer(fm_buf1, W_FC1, B_FC1, fm_fc1_out, FC_IN_SIZE, FC1_OUT_SIZE);
    
    // Apply ReLU to FC1 output
    FC1_RELU: for (int i = 0; i < FC1_OUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        if (fm_fc1_out[i] < 0) fm_fc1_out[i] = 0;
    }
    
    fully_connected_layer(fm_fc1_out, W_FC2, B_FC2, output_fm, FC1_OUT_SIZE, FC_OUT_SIZE);
}
