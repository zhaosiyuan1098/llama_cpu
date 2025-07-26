#include <iostream>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

#include "model.h"
#include "utlis.h"
#include "opt_params.h"
#include "llamaForCausalLM_int4.h"
#include "generate.h"
#include "llamaTokenizer.h"
#include "profiler.h"

// Forward declaration from llamaGenerate.cpp
std::vector<int> BenchmarkLLaMAGenerate(void *model_ptr, int model_type, const std::string& text,
                                        const struct opt_params& generation_config, const std::string& voc_path, bool interactive);

// Copied from main.cpp
std::map<std::string, int> model_config = {{"OPT_125m", OPT_125M}, {"OPT_1.3B", OPT_1_3B}, {"OPT_6.7B", OPT_6_7B}, {"LLaMA_7B", LLaMA_7B}, {"LLaMA_7B_AWQ", LLaMA_7B}, {"LLaMA_7B_2_chat", LLaMA_7B}};

std::map<std::string, std::string> model_path = {
    {"OPT_125m", "models/OPT_125m"},
    {"OPT_1.3B", "models/OPT_1.3B"},
    {"OPT_6.7B", "models/OPT_6.7B"},
    {"LLaMA_7B", "models/LLaMA_7B"},
    {"LLaMA_7B_AWQ", "models/LLaMA_7B_AWQ"},
    {"LLaMA_7B_2_chat", "models/LLaMA_7B_2_chat"},
};

std::map<std::string, int> data_format_list = {
    {"FP32", FP32},
    {"INT8", INT8},
    {"INT4", INT4},
};

bool isLLaMA7B(std::string s)
{
    std::string LLaMA_prefix = "LLaMA_7B";
    return s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix;
}

// The new main function for benchmarking
int main(int argc, char **argv)
{
    std::string target_model = "LLaMA_7B_2_chat";
    std::string target_data_format = "INT4";
    std::cout << "Starting benchmark..." << std::endl;

    if (argc == 3)
    {
        target_model = argv[1];
        target_data_format = argv[2];
    }

    if (model_config.find(target_model) == model_config.end()) {
        std::cerr << "Model config:" << target_model << " unsupported" << std::endl;
        return 1;
    }
    if (data_format_list.find(target_data_format) == data_format_list.end()) {
        std::cerr << "Data format:" << target_data_format << " unsupported" << std::endl;
        return 1;
    }

    std::cout << "Model: " << target_model << ", Data Format: " << target_data_format << std::endl;

    if (isLLaMA7B(target_model))
    {
        int format_id = data_format_list[target_data_format];
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        struct model_config mc = get_opt_model_config(model_id);
        struct opt_params generation_config; // Using default params

        switch (format_id)
        {
        case INT4:
        {
            std::filesystem::path project_root(PROJECT_ROOT_DIR);
            std::filesystem::path m_path_fs = project_root / "model" / "INT4" / model_path[target_model];
            std::string m_path = m_path_fs.string();
            std::cout << "Loading model from path: " << m_path << std::endl;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, mc);
            std::cout << "Model loaded. Starting generation for benchmark." << std::endl;

            std::string input = "A chat between a human and an assistant.\n\n### Human: Tell me a long story about a brave knight.\n### Assistant: \n";
            std::filesystem::path vocab_path_fs = project_root / "model" / "vocab" / "llama_vocab.bin";
            std::string vocab_path = vocab_path_fs.string();
            
            BenchmarkLLaMAGenerate(&model, LLaMA_INT4, input, generation_config, vocab_path, false);
            break;
        }
        default:
        {
            std::cerr << "Only INT4 format is supported for LLaMA7B in this benchmark." << std::endl;
            break;
        }
        }
    }
    return 0;
}


// This is a modified version of LLaMAGenerate from src/llamaGenerate.cpp for benchmarking
std::vector<int> BenchmarkLLaMAGenerate(void *model_ptr, int model_type, const std::string& text,
                               const struct opt_params& generation_config, const std::string& voc_path, bool interactive)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> first_token_time;
    bool first_token_generated = false;

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
    std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    int break_cnt = 2;
    while (n_remain != 0 && break_cnt)
    {
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        if (model_type == LLaMA_INT4)
        {
            auto *model = static_cast<Int4LlamaForCausalLM *>(model_ptr);
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
                model_input = Int4LlamaForCausalLM_input(input_ids_mat);
            }
            
            model_output = model->forward(model_input);
            
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        }
        else
        {
            std::cerr << "Model type not supported" << std::endl;
            exit(1);
        }
        has_past_kv = true;

        const int n_vocab = generation_config.n_vocab;
        std::vector<OPT_token_data> candidates;
        candidates.reserve(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++)
        {
            candidates.emplace_back(OPT_token_data{token_id, logits[token_id], 0.0f});
        }
        OPT_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
        
        // For simplicity, using greedy sampling for benchmark. 
        // You can restore the full sampling logic from the original file if needed.
        int id = sample_token_greedy(&candidates_p);

        // --- BENCHMARKING LOGIC ---
        if (!first_token_generated) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token_generated = true;
        }
        // --- END BENCHMARKING LOGIC ---

        if (id == 2) break; // eos

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive)
            std::cout << llama_id_to_token(vocab, id) << std::flush;

        --n_remain;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (generate_ids.empty()) {
        std::cout << "\nNo tokens generated." << std::endl;
        return generate_ids;
    }

    auto ttft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    size_t num_generated_tokens = generate_ids.size();
    float tpot = 0.0f;
    if (num_generated_tokens > 1) {
        tpot = (total_duration.count() - ttft_duration.count()) / static_cast<float>(num_generated_tokens - 1);
    }

    std::cout << "\n\n--- Benchmark Results ---" << std::endl;
    std::cout << "Total tokens generated: " << num_generated_tokens << std::endl;
    std::cout << "Time to first token (TTFT): " << ttft_duration.count() << " ms" << std::endl;
    std::cout << "Time per output token (TPOT): " << tpot << " ms/token" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "-------------------------" << std::endl;

    return generate_ids;
}
