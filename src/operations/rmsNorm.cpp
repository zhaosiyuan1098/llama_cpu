#include "rmsNorm.h"

#include "operators.h"
#include "utlis.h"

#include <assert.h>
#include <pthread.h>
#include <cmath>

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif

struct rmsnorm_thread_args {
    const Matrix3D<float> *x;
    Matrix3D<float> *output;
    const Matrix3D<float> *weight;
    int start_j;
    int end_j;
    float eps;
};

#ifdef QM_ARM
void* rmsnorm_worker_func_arm(void* args) {
    rmsnorm_thread_args* thread_args = (rmsnorm_thread_args*)args;
    const Matrix3D<float> &x = *(thread_args->x);
    Matrix3D<float> &output = *(thread_args->output);
    const Matrix3D<float> &weight = *(thread_args->weight);
    float eps = thread_args->eps;

    for(int j = thread_args->start_j; j < thread_args->end_j; j++) {
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        for(int k = 0; k < 4096; k += 4) {
            float32x4_t val = vld1q_f32(&x(0, j, k));
            sumv0 = vmlaq_f32(sumv0, val, val);
        }
        float rms_array[4];
        vst1q_f32(rms_array, sumv0);
        float rms = 0.0f;
        for(int i = 0; i < 4; i++) rms += rms_array[i];
        rms = std::sqrt(rms / 4096.0f + eps);

        float32x4_t rms_vec = vdupq_n_f32(rms);
        for(int k = 0; k < 4096; k += 4) {
            float32x4_t val = vld1q_f32(&x(0, j, k));
            float32x4_t w = vld1q_f32(&weight(0, 0, k));
            float32x4_t normalized = vdivq_f32(val, rms_vec);
            float32x4_t out = vmulq_f32(normalized, w);
            vst1q_f32(&output(0, j, k), out);
        }
    }
    return nullptr;
}
#endif

#ifdef QM_x86
void* rmsnorm_worker_func_x86(void* args) {
    rmsnorm_thread_args* thread_args = (rmsnorm_thread_args*)args;
    const Matrix3D<float> &x = *(thread_args->x);
    Matrix3D<float> &output = *(thread_args->output);
    const Matrix3D<float> &weight = *(thread_args->weight);
    float eps = thread_args->eps;

    for(int j = thread_args->start_j; j < thread_args->end_j; j++) {
        __m256 sum = _mm256_setzero_ps();
        for(int k = 0; k < 4096; k += 8) {
            __m256 val = _mm256_loadu_ps(&x(0, j, k));
            sum = _mm256_fmadd_ps(val, val, sum);
        }
        float rms_array[8];
        _mm256_storeu_ps(rms_array, sum);
        float rms = 0.0f;
        for(int i = 0; i < 8; i++) rms += rms_array[i];
        rms = std::sqrt(rms / 4096.0f + eps);

        __m256 rms_vec = _mm256_set1_ps(rms);
        for(int k = 0; k < 4096; k += 8) {
            __m256 val = _mm256_loadu_ps(&x(0, j, k));
            __m256 w = _mm256_loadu_ps(&weight(0, 0, k));
            __m256 normalized = _mm256_div_ps(val, rms_vec);
            __m256 out = _mm256_mul_ps(normalized, w);
            _mm256_storeu_ps(&output(0, j, k), out);
        }
    }
    return nullptr;
}
#endif

void LlamaRMSNorm::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    PROFILE_START(profile_name);
    const int last_dims = 2;

    assert(last_dims == 2);
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);
    assert(x.m_dim_z == weight.m_dim_z);
    assert(x.m_dim_x == 1);
    assert(x.m_dim_z == 4096);

    const int num_threads = 8;
    pthread_t threads[num_threads];
    rmsnorm_thread_args args[num_threads];
    int samples_per_thread = x.m_dim_y / num_threads;

    for(int t = 0; t < num_threads; t++) {
        args[t].x = &x;
        args[t].output = &output;
        args[t].weight = &weight;
        args[t].start_j = t * samples_per_thread;
        args[t].end_j = (t == num_threads -1) ? x.m_dim_y : (t+1) * samples_per_thread;
        args[t].eps = eps;
#ifdef QM_ARM
        pthread_create(&threads[t], nullptr, rmsnorm_worker_func_arm, &args[t]);
#elif defined(QM_x86)
        pthread_create(&threads[t], nullptr, rmsnorm_worker_func_x86, &args[t]);
#endif
    }

    for(int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }

    PROFILE_END(profile_name);
}