#include "llamaDecoder_int4.h"

Int4llamaDecoder::Int4llamaDecoder(std::string param_path, const struct model_config config)
{
    allocate_aligned_memory(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    
    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;

    int max_sqlen = config.max_sqlen;

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->embed_tokens = Embedding(embed_dim, voc_size, padding_idx, embweight);
    load_Embedding_params(this->embed_tokens, param_path + "/embed_tokens");

    // Norm
    Matrix3D<float> norm_weight(new float[embed_dim], 1, 1, embed_dim);
    norm_weight.load((param_path + "/norm/weight.bin").c_str());
    this->norm = LlamaRMSNorm(norm_weight);

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Int4llamaDecoderLayer layer = Int4llamaDecoderLayer(path, config, layer_idx);

        this->layers.push_back(layer);
    }
    std::cout << "Int4llamaDecoder init finished!" << std::endl;
}