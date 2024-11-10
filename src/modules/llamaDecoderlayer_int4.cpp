#include "llamaDecoderlayer_int4.h"


Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {

    this->attn = Int4llamaAttention(param_path + "/self_attn", config);
    std::cout << "Int4llamaDecoderLayer init finished! Layer index: " << layer_idx << std::endl;
    }