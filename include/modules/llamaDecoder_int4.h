#ifndef LLAMA_DECODER
#define LLAMA_DECODER

#include "llamaDecoderlayer_int4.h"
#include "embedding.h"
#include "rmsNorm.h"

struct Int4llamaDecoder_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4llamaDecoder_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    explicit Int4llamaDecoder_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4llamaDecoder_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                           std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(std::move(past_keys_)), past_values(std::move(past_values_)) {
        has_past_keys_values = true;
    }
};

class Int4llamaDecoder {
   public:
    Int4llamaDecoder(std::string param_path, struct model_config config);
    Int4llamaDecoder() = default;
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);
    struct Int4llamaDecoder_output forward(const struct Int4llamaDecoder_input& input);
    Embedding embed_tokens;
    LlamaRMSNorm norm;
    int voc_size{}, embed_dim{}, padding_idx{}, hidden_dim{}, num_heads{};
    std::vector<Int4llamaDecoderLayer> layers;
    std::string profile_name = "Int4llamaDecoder";

   private:
    float* attention_mask_buf{};
    float* pos_embeds_buf{};
    float* last_hidden_states_buf{};
    float* hidden_states_buf{};
};

#endif