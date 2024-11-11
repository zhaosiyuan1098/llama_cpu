#include "llamaAttention_int4.h"
#include <dirent.h>

static float *attn_weights_arr;
static float ***key_states_arr_cache;
static float ***value_states_arr_cache;
static float *attn_output_fp_arr;
static int *cache_num;
static float *query_states_unshape_arr;
static float *attn_output_arr;
static float *attn_output_transpose_arr;
static float *key_states_unshape_arr;
static float *key_states_arr;
static float *value_states_unshape_arr;
static float *value_states_arr;
static float *query_states_arr;
static float *value_states_transpose_arr;

Int4llamaAttention::Int4llamaAttention(std::string param_path, const struct model_config config)
{
    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    std::cout << param_path << std::endl;
    uint8_t *q_weight, *k_weight, *v_weight, *o_weight;
    allocate_aligned_memory(q_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(k_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(v_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(o_weight, (config.embed_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(param_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::cout << "Found file: " << ent->d_name << std::endl;
        }
        closedir(dir);
    } else {
        perror("Could not open directory");
    }
    std::cout << "Parameter path after opendir: " << param_path << std::endl;
    this->q_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(q_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/q_proj");
    this->k_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(k_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/k_proj");
    this->v_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(v_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/v_proj");
    this->o_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(o_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/o_proj");

    float *cos_buf, *sin_buf;
    std::cout<<"max_sqlen: "<<config.max_sqlen<<std::endl;
    std::cout<<"embed_dim: "<<config.embed_dim<<std::endl;
    std::cout<<"num_heads: "<<config.num_heads<<std::endl;
    allocate_aligned_memory(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    allocate_aligned_memory(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    this->rotary_pos_emb = RotaryPosEmb(cos, sin, param_path + "/rotary_emb");

    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F32T(1.0f);
    
}   


void Int4llamaAttention::allocate_memory(const struct model_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    cache_num = new int[config.num_layers];
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    allocate_aligned_memory(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    key_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        key_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }
    value_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        value_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }
}