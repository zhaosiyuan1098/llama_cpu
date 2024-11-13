#include "llamaForCausalLM_int4.h"

Int4LlamaForCausalLM::Int4LlamaForCausalLM(std::string param_path, const struct model_config config)
{
    allocate_aligned_memory(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory(lm_head_weight, (config.embed_dim * config.vocsize * sizeof(uint8_t)) / 2);

    this->decoder = Int4llamaDecoder(param_path + "/decoder", config);
    this->lm_head = Linear_FP_int4(Matrix3D<uint8_t>(lm_head_weight, 1, config.vocsize, config.embed_dim / 2),
                                   param_path + "/lm_head");
    std::cout << "Int4LlamaForCausalLM init finished!" << std::endl;
}

struct Int4LlamaForCausalLM_output Int4LlamaForCausalLM::forward(const struct Int4LlamaForCausalLM_input &input) {
    struct Int4LlamaForCausalLM_output output;
    return output;
}