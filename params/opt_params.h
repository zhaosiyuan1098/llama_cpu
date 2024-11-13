#ifndef OPT_PARAMS
#define OPT_PARAMS
#include <cstdint>
#include<unordered_map>


struct opt_params {
    int32_t seed = -1;        // RNG seed
    int32_t n_threads = 1;    // TODO: fix this
    int32_t n_predict = 128;  // new tokens to predict
    int32_t n_parts = -1;     // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx = 512;      // context size
    int32_t n_batch = 512;    // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep = 0;       // number of tokens to keep from initial prompt
    int32_t n_vocab = 50272;  // vocabulary size

    // sampling parameters
    std::unordered_map<int, float> logit_bias;  // logit bias for specific tokens
    int32_t top_k = 40;                         // <= 0 to use vocab size
    float top_p = 0.95f;                        // 1.0 = disabled
    float tfs_z = 1.00f;                        // 1.0 = disabled
    float typical_p = 1.00f;                    // 1.0 = disabled
    float temp = 0.80f;                         // 1.0 = disabled
    float repeat_penalty = 1.10f;               // 1.0 = disabled
    int32_t repeat_last_n = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float frequency_penalty = 0.00f;            // 0.0 = disabled
    float presence_penalty = 0.00f;             // 0.0 = disabled
    int mirostat = 0;                           // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float mirostat_tau = 5.00f;                 // target entropy
    float mirostat_eta = 0.10f;                 // learning rate
};

#endif