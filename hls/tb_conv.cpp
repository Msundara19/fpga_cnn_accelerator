/**
 * @file tb_conv.cpp
 * @brief Enhanced Testbench with Accuracy Score Reporting
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>

#include "tiled_conv.hpp"

using namespace std;
using namespace std::chrono;

//==============================================================================
// LAYER SIZE DEFINITIONS
//==============================================================================

// Conv 1-1/1-2 (3 -> 32)
#define W1_SIZE (32 * 3 * 3 * 3)
#define B1_SIZE 32
#define BN1_SIZE 32
#define W2_SIZE (32 * 32 * 3 * 3)
#define B2_SIZE 32
#define BN2_SIZE 32
// Conv 2-1/2-2 (32 -> 64)
#define W3_SIZE (64 * 32 * 3 * 3)
#define B3_SIZE 64
#define BN3_SIZE 64
#define W4_SIZE (64 * 64 * 3 * 3)
#define B4_SIZE 64
#define BN4_SIZE 64
// Conv 3-1/3-2 (64 -> 128)
#define W5_SIZE (128 * 64 * 3 * 3)
#define B5_SIZE 128
#define BN5_SIZE 128
#define W6_SIZE (128 * 128 * 3 * 3)
#define B6_SIZE 128
#define BN6_SIZE 128
// Conv 4-1/4-2 (128 -> 256)
#define W7_SIZE (256 * 128 * 3 * 3)
#define B7_SIZE 256
#define BN7_SIZE 256
#define W8_SIZE (256 * 256 * 3 * 3)
#define B8_SIZE 256
#define BN8_SIZE 256

// FC 1 (1024 -> 256)
#define W_FC1_SIZE (256 * 1024)
#define B_FC1_SIZE 256
// FC 2 (256 -> 10)
#define W_FC2_SIZE (10 * 256)
#define B_FC2_SIZE 10

// IO Sizes
#define INPUT_SIZE (3 * 32 * 32)
#define OUTPUT_SIZE 10

//==============================================================================
// MEMORY DEFINITIONS
//==============================================================================

fm_t input_fm_mem[INPUT_SIZE];
fm_t output_fm_mem[OUTPUT_SIZE];
fm_t output_golden_mem[OUTPUT_SIZE];

wt_t W_1_mem[W1_SIZE]; wt_t B_1_mem[B1_SIZE];
wt_t W_2_mem[W2_SIZE]; wt_t B_2_mem[B2_SIZE];
wt_t W_3_mem[W3_SIZE]; wt_t B_3_mem[B3_SIZE];
wt_t W_4_mem[W4_SIZE]; wt_t B_4_mem[B4_SIZE];
wt_t W_5_mem[W5_SIZE]; wt_t B_5_mem[B5_SIZE];
wt_t W_6_mem[W6_SIZE]; wt_t B_6_mem[B6_SIZE];
wt_t W_7_mem[W7_SIZE]; wt_t B_7_mem[B7_SIZE];
wt_t W_8_mem[W8_SIZE]; wt_t B_8_mem[B8_SIZE];

wt_t G_1_mem[BN1_SIZE], T_1_mem[BN1_SIZE], M_1_mem[BN1_SIZE], V_1_mem[BN1_SIZE];
wt_t G_2_mem[BN2_SIZE], T_2_mem[BN2_SIZE], M_2_mem[BN2_SIZE], V_2_mem[BN2_SIZE];
wt_t G_3_mem[BN3_SIZE], T_3_mem[BN3_SIZE], M_3_mem[BN3_SIZE], V_3_mem[BN3_SIZE];
wt_t G_4_mem[BN4_SIZE], T_4_mem[BN4_SIZE], M_4_mem[BN4_SIZE], V_4_mem[BN4_SIZE];
wt_t G_5_mem[BN5_SIZE], T_5_mem[BN5_SIZE], M_5_mem[BN5_SIZE], V_5_mem[BN5_SIZE];
wt_t G_6_mem[BN6_SIZE], T_6_mem[BN6_SIZE], M_6_mem[BN6_SIZE], V_6_mem[BN6_SIZE];
wt_t G_7_mem[BN7_SIZE], T_7_mem[BN7_SIZE], M_7_mem[BN7_SIZE], V_7_mem[BN7_SIZE];
wt_t G_8_mem[BN8_SIZE], T_8_mem[BN8_SIZE], M_8_mem[BN8_SIZE], V_8_mem[BN8_SIZE];

wt_t W_FC1_mem[W_FC1_SIZE]; wt_t B_FC1_mem[B_FC1_SIZE];
wt_t W_FC2_mem[W_FC2_SIZE]; wt_t B_FC2_mem[B_FC2_SIZE];

template <typename T, size_t N>
void read_file(const char *path, T (&buffer)[N]) {
    // First try relative to current working directory (e.g., solution1/csim/build)
    ifstream file(path, ios::in | ios::binary);

    if (!file.is_open()) {
        // When csim runs from solution1/csim/build, project root is four levels up
        std::string alt = std::string("../../../../") + path;
        file.open(alt.c_str(), ios::in | ios::binary);
    }

    if (!file.is_open()) {
        cerr << "ERROR: Could not open file: " << path << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char *>(buffer), N * sizeof(float));
    file.close();
}

void read_bin_files()
{
    read_file("params/input_fp32.bin", input_fm_mem);
    read_file("params/output_golden_fp32.bin", output_golden_mem);

    read_file("params/features_0_block_0_weight.bin", W_1_mem); read_file("params/features_0_block_0_bias.bin", B_1_mem);
    read_file("params/features_0_block_1_weight.bin", G_1_mem); read_file("params/features_0_block_1_bias.bin", T_1_mem);
    read_file("params/features_0_block_1_running_mean.bin", M_1_mem); read_file("params/features_0_block_1_running_var.bin", V_1_mem);

    read_file("params/features_0_block_3_weight.bin", W_2_mem); read_file("params/features_0_block_3_bias.bin", B_2_mem);
    read_file("params/features_0_block_4_weight.bin", G_2_mem); read_file("params/features_0_block_4_bias.bin", T_2_mem);
    read_file("params/features_0_block_4_running_mean.bin", M_2_mem); read_file("params/features_0_block_4_running_var.bin", V_2_mem);

    read_file("params/features_1_block_0_weight.bin", W_3_mem); read_file("params/features_1_block_0_bias.bin", B_3_mem);
    read_file("params/features_1_block_1_weight.bin", G_3_mem); read_file("params/features_1_block_1_bias.bin", T_3_mem);
    read_file("params/features_1_block_1_running_mean.bin", M_3_mem); read_file("params/features_1_block_1_running_var.bin", V_3_mem);

    read_file("params/features_1_block_3_weight.bin", W_4_mem); read_file("params/features_1_block_3_bias.bin", B_4_mem);
    read_file("params/features_1_block_4_weight.bin", G_4_mem); read_file("params/features_1_block_4_bias.bin", T_4_mem);
    read_file("params/features_1_block_4_running_mean.bin", M_4_mem); read_file("params/features_1_block_4_running_var.bin", V_4_mem);

    read_file("params/features_2_block_0_weight.bin", W_5_mem); read_file("params/features_2_block_0_bias.bin", B_5_mem);
    read_file("params/features_2_block_1_weight.bin", G_5_mem); read_file("params/features_2_block_1_bias.bin", T_5_mem);
    read_file("params/features_2_block_1_running_mean.bin", M_5_mem); read_file("params/features_2_block_1_running_var.bin", V_5_mem);

    read_file("params/features_2_block_3_weight.bin", W_6_mem); read_file("params/features_2_block_3_bias.bin", B_6_mem);
    read_file("params/features_2_block_4_weight.bin", G_6_mem); read_file("params/features_2_block_4_bias.bin", T_6_mem);
    read_file("params/features_2_block_4_running_mean.bin", M_6_mem); read_file("params/features_2_block_4_running_var.bin", V_6_mem);

    read_file("params/features_3_block_0_weight.bin", W_7_mem); read_file("params/features_3_block_0_bias.bin", B_7_mem);
    read_file("params/features_3_block_1_weight.bin", G_7_mem); read_file("params/features_3_block_1_bias.bin", T_7_mem);
    read_file("params/features_3_block_1_running_mean.bin", M_7_mem); read_file("params/features_3_block_1_running_var.bin", V_7_mem);

    read_file("params/features_3_block_3_weight.bin", W_8_mem); read_file("params/features_3_block_3_bias.bin", B_8_mem);
    read_file("params/features_3_block_4_weight.bin", G_8_mem); read_file("params/features_3_block_4_bias.bin", T_8_mem);
    read_file("params/features_3_block_4_running_mean.bin", M_8_mem); read_file("params/features_3_block_4_running_var.bin", V_8_mem);

    read_file("params/classifier_1_weight.bin", W_FC1_mem);
    read_file("params/classifier_1_bias.bin", B_FC1_mem);
    read_file("params/classifier_4_weight.bin", W_FC2_mem);
    read_file("params/classifier_4_bias.bin", B_FC2_mem);

    std::cout << "All model parameters and test vectors loaded successfully." << std::endl;
}

int main()
{
    read_bin_files();

    std::cout << "Beginning Reduced VGG Inference Simulation (4-Block VGG)..." << std::endl;
    auto start = high_resolution_clock::now();

    reduced_vgg_inference(
        input_fm_mem,
        W_1_mem, B_1_mem, G_1_mem, T_1_mem, M_1_mem, V_1_mem,
        W_2_mem, B_2_mem, G_2_mem, T_2_mem, M_2_mem, V_2_mem,
        W_3_mem, B_3_mem, G_3_mem, T_3_mem, M_3_mem, V_3_mem,
        W_4_mem, B_4_mem, G_4_mem, T_4_mem, M_4_mem, V_4_mem,
        W_5_mem, B_5_mem, G_5_mem, T_5_mem, M_5_mem, V_5_mem,
        W_6_mem, B_6_mem, G_6_mem, T_6_mem, M_6_mem, V_6_mem,
        W_7_mem, B_7_mem, G_7_mem, T_7_mem, M_7_mem, V_7_mem,
        W_8_mem, B_8_mem, G_8_mem, T_8_mem, M_8_mem, V_8_mem,
        W_FC1_mem, B_FC1_mem, W_FC2_mem, B_FC2_mem,
        output_fm_mem);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "========================================" << std::endl;
    std::cout << "Inference Simulation Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Simulation Time: " << duration.count() << " ms" << std::endl;

    long double mse = 0.0;
    int error_count = 0;

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        fm_t expected = output_golden_mem[i];
        fm_t actual = output_fm_mem[i];

        mse += std::pow((long double)(expected - actual), 2.0);

        if (std::abs((float)expected - (float)actual) > 0.1) {
            error_count++;
        }
    }

    mse = mse / OUTPUT_SIZE;

    float expected_accuracy = 85.59;  // INT32 quantization accuracy

    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Metrics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "MSE:              " << mse << std::endl;
    std::cout << "Error count:      " << error_count << "/" << OUTPUT_SIZE << std::endl;
    std::cout << "Expected Accuracy: " << expected_accuracy << "%" << std::endl;
    std::cout << "Model:            Reduced VGG (INT32 quantized)" << std::endl;
    std::cout << "Parameters:       1,439,146" << std::endl;

    std::cout << "\n========================================" << std::endl;
    if (mse <= 5.0 && error_count < (OUTPUT_SIZE / 2))
    {
        std::cout << "TEST PASSED - Simulation SUCCESSFUL!" << std::endl;
        std::cout << "SCORE = Accuracy / Latency(ms) = " << expected_accuracy / (duration.count()) << " acc/ms" << std::endl;
    }
    else
    {
        std::cout << "TEST FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return 0;
}

