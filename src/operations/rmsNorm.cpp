#include "rmsNorm.h"

#include "operators.h"
#include "utlis.h"
#include "common.h"
#include "assert.h"
#include <math.h>
void LlamaRMSNorm::forward(const Matrix3D<float> &x, Matrix3D<float> &output)
{
    PROFILE_START(profile_name);
    const int last_dims = 2;

    assert(last_dims == 2); // support the last dim for now
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);
    assert(x.m_dim_z == weight.m_dim_z);

    for (int i = 0; i < x.m_dim_x; i++)
    { // batches
        for (int j = 0; j < x.m_dim_y; j++)
        { // samples
            float rms = 0;

            // Step 1: Compute RMS
            for (int k = 0; k < x.m_dim_z; k++)
            { // hidden states
                rms += x(i, j, k) * x(i, j, k);
            }
            rms = sqrt(rms / static_cast<float>(x.m_dim_z) + eps);

            // Step 2: Normalize and apply the weight
            for (int k = 0; k < x.m_dim_z; k++)
            {
                float value = static_cast<float>(x(i, j, k));
                float fp_out = (value / rms) * weight(0, 0, k); // Apply normalization and scaling
                output(i, j, k) = fp_out;
            }
        }
    }

    PROFILE_END(profile_name);
}