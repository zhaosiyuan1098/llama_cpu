#include"embedding.h"

void load_Embedding_params(Embedding& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
    // read_to_array((prefix + "/weight.bin").c_str(), op.lookup.m_data, op.lookup.length());
}