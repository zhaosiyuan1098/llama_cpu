#include"llamaAttention.h"


Fp32llamaAttention::Fp32llamaAttention(std::string param_path, const struct model_config config) {
    float *q_weight, *k_weight, *v_weight, *o_weight;
    // allocate_aligned_memory(q_weight, config.embed_dim * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(k_weight, config.embed_dim * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(v_weight, config.embed_dim * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(o_weight, config.embed_dim * config.embed_dim * sizeof(float));
    // this->q_proj =
    //     Linear_FP(Matrix3D<float>(q_weight, 1, config.embed_dim, config.embed_dim), param_path + "/q_proj/weight.bin");
    // this->k_proj =
    //     Linear_FP(Matrix3D<float>(k_weight, 1, config.embed_dim, config.embed_dim), param_path + "/k_proj/weight.bin");
    // this->v_proj =
    //     Linear_FP(Matrix3D<float>(v_weight, 1, config.embed_dim, config.embed_dim), param_path + "/v_proj/weight.bin");
    // this->o_proj =
    //     Linear_FP(Matrix3D<float>(o_weight, 1, config.embed_dim, config.embed_dim), param_path + "/o_proj/weight.bin");

    // float *cos_buf, *sin_buf;
    // allocate_aligned_memory(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    // allocate_aligned_memory(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    // Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    // Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    // this->rotary_pos_emb = RotaryPosEmb(cos, sin, param_path + "/rotary_emb");

    // float qk_bmm_alpha;
    // read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    // this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    // this->pv_bmm = BMM_F32T(1.0f);

    // this->embed_dim = config.embed_dim;
    // this->num_heads = config.num_heads;
    // assert(config.embed_dim % config.num_heads == 0);
    // this->head_dim = config.embed_dim / config.num_heads;
}