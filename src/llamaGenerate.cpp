#include "common.h"
#include "utlis.h"
#include "generate.h"
#include "llamaTokenizer.h"

std::vector<int> LLaMAGenerate(void *model_ptr, int model_type, std::string text,
                               const struct opt_params generation_config, std::string voc_path, bool interactive)
{
    std::vector<int> ans{};
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
    input_ids.resize(n);

    int n_consumed = 0;
    while ((int)input_ids.size() > n_consumed)
    {
        embd.push_back(input_ids[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(input_ids[n_consumed]);
        ++n_consumed;

        if ((int)embd.size() >= generation_config.n_batch)
        {
            break;
        }
    }

    if (interactive)
        std::cout << "ASSISTANT: " << std::endl;

    bool has_past_kv = false;
    bool previous_two_hash = false;
    std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    int break_cnt = 2;
    int input_size = input_ids.size();

    while (n_remain && break_cnt)
    {
        std::vector<float> logits(generation_config.n_vocab);
        int sqlen = 1;
        if (model_type == LLaMA_INT4)
        {
            Int4LlamaForCausalLM *model = static_cast<Int4LlamaForCausalLM *>(model_ptr);
            struct Int4LlamaForCausalLM_output model_output;
            struct Int4LlamaForCausalLM_input model_input;
            if (has_past_kv)
            {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            }
            else
            {
                sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat};
            }
            if (has_past_kv)
                STATS_START("Inference latency");
            model_output = model->forward(model_input);
            if (has_past_kv)
                STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
            has_past_kv = true;
        }
        else
        {
            std::cerr << "Model type currently not supported" << std::endl;
            break;
        }
        const int n_ctx = generation_config.n_ctx;
        const float temp = generation_config.temp;
        const int32_t top_k = generation_config.top_k <= 0 ? generation_config.n_vocab : generation_config.top_k;
        const float top_p = generation_config.top_p;
        const float tfs_z = generation_config.tfs_z;
        const float typical_p = generation_config.typical_p;
        const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? n_ctx : generation_config.repeat_last_n;
        const float repeat_penalty = generation_config.repeat_penalty;
        const float alpha_presence = generation_config.presence_penalty;
        const float alpha_frequency = generation_config.frequency_penalty;
        const int mirostat = generation_config.mirostat;
        const float mirostat_tau = generation_config.mirostat_tau;
        const float mirostat_eta = generation_config.mirostat_eta;
        const int n_vocab = generation_config.n_vocab;
    }

    if (interactive)
    {
        std::cout << "USER: ";
        std::string input;
        std::getline(std::cin, input);
        input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
        // LLaMAGenerate(model_ptr, model_type, input, generation_config, voc_path, false);
    }
    std::cout << "finished" << std::endl;
    return ans;
}
