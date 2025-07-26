#include <iostream>
#include <filesystem>
#include <map>
#include "model.h"
#include "utlis.h"
#include "opt_params.h"
#include "llamaForCausalLM_int4.h"
#include "generate.h"


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

    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix)
        return true;
    else
        return false;
}

int main(int argc, char **argv)
{
    std::string target_model = "LLaMA_7B_2_chat";
    std::string target_data_format = "INT4";
    std::cout << "hello! I am yuanigne, powered by @zhaosiyuan1098. " << std::endl;

    if (argc == 3)
    {
        // 模型参数处理
        auto target_str = argv[1];
        if (model_config.count(target_model) == 0)
        {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto &k : model_config)
            {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Model: " << argv[1] << " selected" << std::endl;
        target_model = argv[1];
        auto data_format_input = argv[2];
        if (data_format_list.count(data_format_input) == 0)
        {
            std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto &k : data_format_list)
            {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported data format\n");
            std::cout << "Data format: " << argv[2] << " selected" << std::endl;
            target_data_format = argv[2];
        }
        else
        {
            if (isLLaMA7B(target_model))
            {
                std::cout << "Using model: " + target_model << std::endl;
                std::cout << "Using LLaMA's default data format: " + target_data_format << std::endl;
            }
            else
            { // OPT
                target_model = "OPT6.7B";
                target_data_format = "INT8";
                std::cout << "Using model: " + target_model << std::endl;
                std::cout << "Using OPT's default data format: " + target_data_format << std::endl;
            }
        }
    }

    if (isLLaMA7B(target_model))
    {
        int format_id = data_format_list[target_data_format];
        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];
        struct model_config model_config = get_opt_model_config(model_id);
        struct opt_params generation_config;
        
        // generation_config.n_predict = 512;
        // generation_config.n_vocab = 32000;
        // generation_config.temp = 0.1f;
        // generation_config.repeat_penalty = 1.25f;

        switch (format_id)
        {
        case INT4:
        {
            std::filesystem::path project_root(PROJECT_ROOT_DIR); // 直接使用宏
            std::filesystem::path m_path_fs = project_root / "model" / "INT4" / model_path[target_model];
            std::string m_path = m_path_fs.string();
            std::cout << "Loading model from path: " << m_path << std::endl;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, model_config);
            std::cout << "All load Finished! now you can chat with llm in the terminal~" << std::endl;

            // Get input from the user
                auto i=0;
            while (i<1)
            {
                std::cout << "USER: ";
                std::string input="how to get a cup of tea";
                // std::getline(std::cin, input);
                input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
                std::filesystem::path vocab_path_fs = project_root / "model" / "vocab" / "llama_vocab.bin";
                std::string vocab_path = vocab_path_fs.string();
                LLaMAGenerate(&model, LLaMA_INT4, input, generation_config, vocab_path, true);
                i++;
            }
            break;
        }
        default:
        {
            std::cout << std::endl;
            std::cerr << "only support INT4 for LLaMA7B." << std::endl;
            break;
        }
        }
    }
    return 0;
}
