

#include"linear.h"

#include "operators.h"
#include "utlis.h"
#include "matmul.h"

static int8_t *x_int8;
static float *x_scale;

template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c) {
    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

    int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;

    for (int b_ = 0; b_ < b_size; b_++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T acc = 0;
                for (int kk = 0; kk < k; k++) {
                    acc += a(b_, i, kk) * b(b_, j, kk);
                }

                c(b_, i, j) = acc;
            }
        }
    }
}

void Linear_FP_int4::initialize_memory(const int block_size) {
    allocate_aligned_memory(x_int8, MAX_LINEAR_LENGTH * sizeof(int8_t));
    allocate_aligned_memory(x_scale, (MAX_LINEAR_LENGTH / block_size) * sizeof(float));
}

void Linear_FP_int4::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    const int num_thread = 16;
    Matrix3D<uint8_t> b = this->weight;
    const int m = x.m_dim_y, n = b.m_dim_y, k = x.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(output.m_dim_x == 1);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == weight.m_dim_y);
    assert(x.m_dim_z / 2 == weight.m_dim_z);

    assert(output.m_dim_z > num_thread);
    assert(output.m_dim_z % (num_thread * 2) == 0);  // unroll column by 2

    struct matmul_params params;
    params.A.row = x.m_dim_y;
    params.A.column = x.m_dim_z;
    params.A.data_ptr = x.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.int4_data_ptr = b.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;
    params.opt_params.num_thread = num_thread;
    params.scales = this->scale.m_data;
    params.offset = this->offset.m_data;
    params.block_size = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    if (!x_int8) this->initialize_memory(params.block_size);
    params.A.int8_data_ptr = x_int8;
    params.A_scales = x_scale;

#if IMP == 0
    op.mat_mul_reference(&params);
#elif IMP == 1
    op.mat_mul_loop_unrolling(&params);
#elif IMP == 2
    op.mat_mul_multithreading(&params);
#elif IMP == 3
    op.mat_mul_simd_programming(&params);
#elif IMP == 4
    op.mat_mul_all_techniques(&params);
#else
    printf("Implementation not specified\n");
#endif
    PROFILE_END(profile_name);
    return;
}
