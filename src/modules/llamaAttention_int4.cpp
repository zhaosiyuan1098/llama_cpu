#include "llamaAttention_int4.h"
#include <dirent.h>

Int4llamaAttention::Int4llamaAttention(std::string param_path, const struct model_config config)
{
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
}