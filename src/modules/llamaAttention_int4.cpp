#include "llamaAttention_int4.h"

Int4llamaAttention::Int4llamaAttention(std::string param_path, const struct model_config config)
{
    std::cout << param_path << std::endl;
    uint8_t *q_weight, *k_weight, *v_weight, *o_weight;
    allocate_aligned_memory(q_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(k_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(v_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(o_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    // this->q_proj =
    //     Linear_FP_int4(Matrix3D<uint8_t>(q_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/q_proj");
    std::cout << "Allocated memory" << std::endl;
}