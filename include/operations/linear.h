#ifndef LINEAR_H
#define LINEAR_H

#include "common.h"
#include"utlis.h"
#include "assert.h"

class Linear_FP {
   public:
    Linear_FP(Matrix3D<float> weight_, std::string weight_path) : weight(weight_) {
        read_to_array((weight_path).c_str(), this->weight.m_data, this->weight.length());
    };
    Linear_FP()= default;
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;

   private:
    std::string profile_name = "Linear_FP";
};

class Linear_FP_int4 {
   public:
    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path) : weight(weight_) {
        float *scale_ptr, *zero_point_ptr;
        float *offset_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        assert((weight.m_dim_z * 2) % (QK) == 0);
        allocate_aligned_memory(scale_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(offset_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(zero_point_ptr, 1 * sizeof(float));

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
        scale = Matrix3D<float>(scale_ptr, x, y, z);
        offset = Matrix3D<float>(offset_ptr, x, y, z);
        zero_point = Matrix3D<float>(zero_point_ptr, 1, 1, 1);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        offset.load((weight_path + "/offset_int4.bin").c_str());
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_FP_int4(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_ref(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output);
    static void initialize_memory(const int block_size);
    Matrix3D<uint8_t> weight;
    Matrix3D<float> scale, zero_point;
    Matrix3D<float> offset;

#if IMP == 0
    std::string profile_name = "reference";
#elif IMP == 1
    std::string profile_name = "loop_unrolling";
#elif IMP == 2
    std::string profile_name = "multithreading";
#elif IMP == 3
    std::string profile_name = "simd_programming";
#elif IMP == 4
    std::string profile_name = "multithreading_loop_unrolling";
#elif IMP == 5
    std::string profile_name = "all_techniques";
#else
    std::string profile_name = "Unkown";
#endif
};

#endif // LINEAR_H