#ifndef BMM_H
#define BMM_H

#include"common.h"

class BMM_F32T {
   public:
    explicit BMM_F32T(float _alpha);
    BMM_F32T()= default;
    void forward(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    void forward_weight_untransposed(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    float alpha{};

   private:
    std::string profile_name = "BMM_F32T";
};

#endif