#include"common.h"
#include"utlis.h"

class RotaryPosEmb
{
public:
    RotaryPosEmb(Matrix3D<float> _cos, Matrix3D<float> _sin, std::string path)
    {
        sin = _sin;
        cos = _cos;
        read_to_array((path + "/cos_cached.bin").c_str(), cos.m_data, cos.length());
        read_to_array((path + "/sin_cached.bin").c_str(), sin.m_data, sin.length());
    };
    RotaryPosEmb()= default;
    void forward(Matrix3D<float> &key, Matrix3D<float> &value, int start_idx, int len);
    Matrix3D<float> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb";
};