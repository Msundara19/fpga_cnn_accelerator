/**
 * @file utils.cpp - RESOURCE-CONSTRAINED VERSION
 * @brief Optimized for Zynq-7020 resource limits
 * Target: < 80% resource utilization while maintaining good latency
 */

#include "tiled_conv.hpp"

//==============================================================================
// LOAD INPUT TILE FROM DRAM
//==============================================================================
void load_input_tile_from_DRAM(
    fm_t *in_fm,
    fm_t in_fm_buf[MAX_M][MAX_IN_BUF_HW][MAX_IN_BUF_HW],
    int tr, int tc,
    int M, int I, int P, int TILE_HW, int MARGIN)
{
    #pragma HLS INLINE OFF
    
    LOAD_M: for (int m = 0; m < M; m++) {
        LOAD_R: for (int r = 0; r < TILE_HW + 2*MARGIN; r++) {
            #pragma HLS PIPELINE II=1
            LOAD_C: for (int c = 0; c < TILE_HW + 2*MARGIN; c++) {
                int global_r = tr * TILE_HW + r - P;
                int global_c = tc * TILE_HW + c - P;
                
                if (global_r >= 0 && global_r < I && global_c >= 0 && global_c < I) {
                    in_fm_buf[m][r][c] = in_fm[m * I * I + global_r * I + global_c];
                } else {
                    in_fm_buf[m][r][c] = 0;
                }
            }
        }
    }
}

//==============================================================================
// LOAD LAYER PARAMETERS FROM DRAM
//==============================================================================
void load_layer_params_from_DRAM(
    wt_t *weights,
    wt_t *biases,
    wt_t weights_buf[MAX_OUT_BUF_DEPTH][MAX_M][K][K],
    wt_t biases_buf[MAX_OUT_BUF_DEPTH],
    int kernel_group,
    int M, int N, int OUT_BUF_DEPTH)
{
    #pragma HLS INLINE OFF
    
    LOAD_BIAS: for (int n = 0; n < OUT_BUF_DEPTH; n++) {
        #pragma HLS PIPELINE II=1
        biases_buf[n] = biases[kernel_group * OUT_BUF_DEPTH + n];
    }
    
    LOAD_N: for (int n = 0; n < OUT_BUF_DEPTH; n++) {
        LOAD_M: for (int m = 0; m < M; m++) {
            LOAD_KR: for (int kr = 0; kr < K; kr++) {
                #pragma HLS PIPELINE II=1
                LOAD_KC: for (int kc = 0; kc < K; kc++) {
                    int w_idx = ((kernel_group * OUT_BUF_DEPTH + n) * M + m) * K * K + kr * K + kc;
                    weights_buf[n][m][kr][kc] = weights[w_idx];
                }
            }
        }
    }
}

//==============================================================================
// 3x3 CONVOLUTION - REDUCED PARALLELISM
//==============================================================================
void conv_3x3(
    fm_t X_buf[MAX_M][MAX_IN_BUF_HW][MAX_IN_BUF_HW],
    wt_t W_buf[MAX_OUT_BUF_DEPTH][MAX_M][K][K],
    wt_t B_buf[MAX_OUT_BUF_DEPTH],
    fm_t Y_buf[MAX_OUT_BUF_DEPTH][MAX_TILE_HW][MAX_TILE_HW],
    int M, int TILE_HW, int OUT_BUF_DEPTH)
{
    #pragma HLS INLINE OFF
    
    CONV_N: for (int n = 0; n < OUT_BUF_DEPTH; n++) {
        CONV_R: for (int r = 0; r < TILE_HW; r++) {
            CONV_C: for (int c = 0; c < TILE_HW; c++) {
                #pragma HLS PIPELINE II=1
                
                acc_t sum = (acc_t)B_buf[n];
                
                CONV_M: for (int m = 0; m < M; m++) {
                    CONV_KR: for (int kr = 0; kr < K; kr++) {
                        CONV_KC: for (int kc = 0; kc < K; kc++) {
                            // Removed UNROLL - reduces DSP usage significantly
                            sum += (acc_t)X_buf[m][r + kr][c + kc] * (acc_t)W_buf[n][m][kr][kc];
                        }
                    }
                }
                
                Y_buf[n][r][c] = (fm_t)sum;
            }
        }
    }
}

//==============================================================================
// STORE OUTPUT TILE TO DRAM
//==============================================================================
void store_output_tile_to_DRAM(
    fm_t Y_buf[MAX_OUT_BUF_DEPTH][MAX_TILE_HW][MAX_TILE_HW],
    fm_t *out_fm,
    int tr, int tc, int kernel_group,
    int N, int O, int TILE_HW, int OUT_BUF_DEPTH)
{
    #pragma HLS INLINE OFF
    
    STORE_N: for (int n = 0; n < OUT_BUF_DEPTH; n++) {
        STORE_R: for (int r = 0; r < TILE_HW; r++) {
            #pragma HLS PIPELINE II=1
            STORE_C: for (int c = 0; c < TILE_HW; c++) {
                int global_r = tr * TILE_HW + r;
                int global_c = tc * TILE_HW + c;
                int global_n = kernel_group * OUT_BUF_DEPTH + n;
                
                if (global_r < O && global_c < O && global_n < N) {
                    out_fm[global_n * O * O + global_r * O + global_c] = Y_buf[n][r][c];
                }
            }
        }
    }
}

//==============================================================================
// BATCH NORMALIZATION + ReLU
//==============================================================================
void batch_norm_relu_layer(
    fm_t *input_fm,
    wt_t *gamma, wt_t *beta, wt_t *mean, wt_t *variance,
    fm_t *output_fm,
    int M, int I)
{
    #pragma HLS INLINE OFF
    
    BN_M: for (int m = 0; m < M; m++) {
        #pragma HLS LOOP_TRIPCOUNT min=32 max=256
        
        acc_t std_val = hls::sqrt((acc_t)variance[m] + (acc_t)BN_EPSILON);
        
        BN_R: for (int r = 0; r < I; r++) {
            #pragma HLS PIPELINE II=1
            BN_C: for (int c = 0; c < I; c++) {
                int addr = m * I * I + r * I + c;
                fm_t input_val = input_fm[addr];
                
                acc_t normalized = ((acc_t)input_val - (acc_t)mean[m]) / std_val;
                acc_t scaled = (acc_t)gamma[m] * normalized + (acc_t)beta[m];
                
                output_fm[addr] = (scaled > 0) ? (fm_t)scaled : (fm_t)0;
            }
        }
    }
}

//==============================================================================
// MAX POOLING 2x2
//==============================================================================
void max_pool_2x2(
    fm_t *input_fm,
    fm_t *output_fm,
    int M, int I, int O)
{
    #pragma HLS INLINE OFF
    
    POOL_M: for (int m = 0; m < M; m++) {
        #pragma HLS LOOP_TRIPCOUNT min=32 max=256
        
        POOL_R: for (int r = 0; r < O; r++) {
            #pragma HLS PIPELINE II=2
            POOL_C: for (int c = 0; c < O; c++) {
                int in_r = r * 2;
                int in_c = c * 2;
                
                fm_t max_val = input_fm[m * I * I + in_r * I + in_c];
                
                POOL_WINDOW: for (int wr = 0; wr < 2; wr++) {
                    for (int wc = 0; wc < 2; wc++) {
                        // Removed UNROLL
                        fm_t val = input_fm[m * I * I + (in_r + wr) * I + (in_c + wc)];
                        if (val > max_val) max_val = val;
                    }
                }
                
                output_fm[m * O * O + r * O + c] = max_val;
            }
        }
    }
}

//==============================================================================
// FULLY CONNECTED LAYER - SEQUENTIAL (LOW RESOURCE)
//==============================================================================
void fully_connected_layer(
    fm_t *input_fm,
    wt_t *weights,
    wt_t *biases,
    fm_t *output_fm,
    int IN_SIZE, int OUT_SIZE)
{
    #pragma HLS INLINE OFF
    
    FC_OUT: for (int o = 0; o < OUT_SIZE; o++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=256
        
        acc_t sum = (acc_t)biases[o];
        
        FC_IN: for (int i = 0; i < IN_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            sum += (acc_t)input_fm[i] * (acc_t)weights[o * IN_SIZE + i];
        }
        
        output_fm[o] = (fm_t)sum;
    }
}

//==============================================================================
// TILED CONVOLUTION LAYER - REDUCED ARRAY PARTITIONING
//==============================================================================
void tiled_conv_layer(
    fm_t *input_fm,
    wt_t *weights,
    wt_t *biases,
    fm_t *output_fm,
    int M, int I, int N, int O, int S, int P,
    int TILE_HW, int OUT_BUF_DEPTH)
{
    #pragma HLS INLINE OFF
    
    fm_t in_fm_buf[MAX_M][MAX_IN_BUF_HW][MAX_IN_BUF_HW];
    wt_t weights_buf[MAX_OUT_BUF_DEPTH][MAX_M][K][K];
    wt_t biases_buf[MAX_OUT_BUF_DEPTH];
    fm_t out_buf[MAX_OUT_BUF_DEPTH][MAX_TILE_HW][MAX_TILE_HW];
    
    // REDUCED partitioning to save resources
    #pragma HLS ARRAY_PARTITION variable=in_fm_buf cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=weights_buf cyclic factor=2 dim=4
    #pragma HLS ARRAY_PARTITION variable=out_buf cyclic factor=2 dim=1
    
    int num_tiles = (O + TILE_HW - 1) / TILE_HW;
    int num_kernel_groups = (N + OUT_BUF_DEPTH - 1) / OUT_BUF_DEPTH;
    
    TILE_ROW: for (int tr = 0; tr < num_tiles; tr++) {
        TILE_COL: for (int tc = 0; tc < num_tiles; tc++) {
            
            load_input_tile_from_DRAM(input_fm, in_fm_buf, tr, tc, M, I, P, TILE_HW, P);
            
            KERNEL_GROUP: for (int kg = 0; kg < num_kernel_groups; kg++) {
                
                load_layer_params_from_DRAM(weights, biases, weights_buf, biases_buf, kg, M, N, OUT_BUF_DEPTH);
                
                conv_3x3(in_fm_buf, weights_buf, biases_buf, out_buf, M, TILE_HW, OUT_BUF_DEPTH);
                
                store_output_tile_to_DRAM(out_buf, output_fm, tr, tc, kg, N, O, TILE_HW, OUT_BUF_DEPTH);
            }
        }
    }
}
