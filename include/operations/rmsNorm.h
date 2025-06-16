#ifndef RMSNORM_H
#define RMSNORM_H

#include"common.h"

class LlamaRMSNorm {
   public:
    explicit LlamaRMSNorm(Matrix3D<float> _weight) : weight(_weight){};
    LlamaRMSNorm()= default;
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;
    float eps = 1e-6;

   private:
    std::string profile_name = "LlamaRMSNorm";
};

#endif