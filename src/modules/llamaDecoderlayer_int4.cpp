#include "llamaDecoderlayer_int4.h"


static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;
static float *temp;
static float *hidden_states_arr;


Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if(layer_idx==0){
        this->allocate_memory(config);
        Int4llamaAttention::allocate_memory(config);
    }

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4llamaAttention(param_path + "/self_attn", config);

    float *input_layernorm_weight_ptr;
    allocate_aligned_memory(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    this->input_layernorm = LlamaRMSNorm(input_layernorm_weight);

    float *post_attention_layernorm_ptr;
    allocate_aligned_memory(post_attention_layernorm_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> post_attention_layernorm_weight(post_attention_layernorm_ptr, 1, 1, config.embed_dim);
    post_attention_layernorm_weight.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    this->post_attention_layernorm = LlamaRMSNorm(post_attention_layernorm_weight);


    uint8_t *gate_proj_weight, *down_proj_weight, *up_proj_weight;
    allocate_aligned_memory(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t)) / 2);
    this->gate_proj = Linear_FP_int4(Matrix3D<uint8_t>(gate_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                     (param_path + "/gate_proj"));
    this->down_proj = Linear_FP_int4(Matrix3D<uint8_t>(down_proj_weight, 1, config.embed_dim, config.hidden_dim / 2),
                                     (param_path + "/down_proj"));
    this->up_proj = Linear_FP_int4(Matrix3D<uint8_t>(up_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                   (param_path + "/up_proj"));
    std::cout << "Int4llamaDecoderLayer init finished! Layer index: " << layer_idx << std::endl;
    }

void Int4llamaDecoderLayer::allocate_memory(const struct model_config config) {
    allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(gate_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(up_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(temp, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
}