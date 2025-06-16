#include <cassert>
#include <pthread.h>
#include <cstdio>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "opt_params.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif

struct w4a8_thread_args
{
    int start_j, end_j;
    const struct matmul_params *params;
};

static void *all_techniques_worker_func(void *args)
{
    auto *mat_args = (struct w4a8_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    // const int num_block = k / block_size; // block_size = 32

    for (int row = 0; row < m; row++)
    {
        for (int col = mat_args->start_j; col < mat_args->end_j; col++)
        {
#ifdef QM_ARM
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            const unsigned char *w_start = &params->B.int4_data_ptr[col * k / 2];
            const signed char *a_start = &params->A.int8_data_ptr[row * k];
            float *s_a = &params->A_scales[row * k / 32];
            float *s_w = &params->scales[col * k / 32];

            for (int q = 0; q < num_block; q += 4)
            {
                const uint8x16_t w0 = vld1q_u8(w_start);
                const uint8x16_t w1 = vld1q_u8(w_start + 16);
                const uint8x16_t w2 = vld1q_u8(w_start + 32);
                const uint8x16_t w3 = vld1q_u8(w_start + 48);
                w_start += 64;

                const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                const int8x16_t offsets = vdupq_n_s8(8);

                // 解码权重
                uint8x16_t low_half0 = vandq_u8(w0, mask_low4bit);
                uint8x16_t high_half0 = vshrq_n_u8(w0, 4);
                high_half0 = vandq_u8(high_half0, mask_low4bit);

                uint8x16_t low_half1 = vandq_u8(w1, mask_low4bit);
                uint8x16_t high_half1 = vshrq_n_u8(w1, 4);
                high_half1 = vandq_u8(high_half1, mask_low4bit);

                uint8x16_t low_half2 = vandq_u8(w2, mask_low4bit);
                uint8x16_t high_half2 = vshrq_n_u8(w2, 4);
                high_half2 = vandq_u8(high_half2, mask_low4bit);

                uint8x16_t low_half3 = vandq_u8(w3, mask_low4bit);
                uint8x16_t high_half3 = vshrq_n_u8(w3, 4);
                high_half3 = vandq_u8(high_half3, mask_low4bit);
                int8x16_t low_half_s8_0 = vreinterpretq_s8_u8(low_half0);
                int8x16_t high_half_s8_0 = vreinterpretq_s8_u8(high_half0);

                int8x16_t low_half_s8_1 = vreinterpretq_s8_u8(low_half1);
                int8x16_t high_half_s8_1 = vreinterpretq_s8_u8(high_half1);

                int8x16_t low_half_s8_2 = vreinterpretq_s8_u8(low_half2);
                int8x16_t high_half_s8_2 = vreinterpretq_s8_u8(high_half2);

                int8x16_t low_half_s8_3 = vreinterpretq_s8_u8(low_half3);
                int8x16_t high_half_s8_3 = vreinterpretq_s8_u8(high_half3);

                // 应用零点偏移
                low_half_s8_0 = vsubq_s8(low_half_s8_0, offsets);
                high_half_s8_0 = vsubq_s8(high_half_s8_0, offsets);

                low_half_s8_1 = vsubq_s8(low_half_s8_1, offsets);
                high_half_s8_1 = vsubq_s8(high_half_s8_1, offsets);

                low_half_s8_2 = vsubq_s8(low_half_s8_2, offsets);
                high_half_s8_2 = vsubq_s8(high_half_s8_2, offsets);

                low_half_s8_3 = vsubq_s8(low_half_s8_3, offsets);
                high_half_s8_3 = vsubq_s8(high_half_s8_3, offsets);

                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // 计算点积
                int32x4_t int_sum0 = vdotq_s32(vdupq_n_s32(0), a0, low_half_s8_0);
                int_sum0 = vdotq_s32(int_sum0, a1, high_half_s8_0);

                int32x4_t int_sum1 = vdotq_s32(vdupq_n_s32(0), a2, low_half_s8_1);
                int_sum1 = vdotq_s32(int_sum1, a3, high_half_s8_1);

                int32x4_t int_sum2 = vdotq_s32(vdupq_n_s32(0), a4, low_half_s8_2);
                int_sum2 = vdotq_s32(int_sum2, a5, high_half_s8_2);

                int32x4_t int_sum3 = vdotq_s32(vdupq_n_s32(0), a6, low_half_s8_3);
                int_sum3 = vdotq_s32(int_sum3, a7, high_half_s8_3);

                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
            }
            sumv0 = vaddq_f32(sumv0, sumv1);
            sumv2 = vaddq_f32(sumv2, sumv3);
            sumv0 = vaddq_f32(sumv0, sumv2);
            params->C.data_ptr[row * n + col] = vaddvq_f32(sumv0);

#endif
#ifdef QM_x86
            __m256 accumulator = _mm256_setzero_ps();
            float *s_ptr = &params->scales[col * k / 32];
            float *sa_ptr = &params->A_scales[row * k / 32];
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            const int num_block = k / block_size;
            const __m256i lowMask = _mm256_set1_epi8(0xF);

            for (int q = 0; q < num_block; q += 4)
            {
                

                __m256i raw_w = _mm256_loadu_si256(w_start);
                __m256i raw_w_next = _mm256_loadu_si256(w_start + 1);

                __m256i low_half_weight = _mm256_and_si256(raw_w, lowMask);
                __m256i shifted_w = _mm256_srli_epi16(raw_w, 4);
                __m256i high_half_weight = _mm256_and_si256(shifted_w, lowMask);

                __m256i low_half_weight_next = _mm256_and_si256(raw_w_next, lowMask);
                __m256i shifted_w_next = _mm256_srli_epi16(raw_w_next, 4);
                __m256i high_half_weight_next = _mm256_and_si256(shifted_w_next, lowMask);

                const __m256i zero_point = _mm256_set1_epi8(8);
                __m256i w_0, w_128, w_0_next, w_128_next;

                w_0 = _mm256_sub_epi8(low_half_weight, zero_point);
                w_128 = _mm256_sub_epi8(high_half_weight, zero_point);

                w_0_next = _mm256_sub_epi8(low_half_weight_next, zero_point);
                w_128_next = _mm256_sub_epi8(high_half_weight_next, zero_point);

                __m256i dot, dot2, dot3, dot4;
                const __m256i ax = _mm256_sign_epi8(w_0, w_0);
                const __m256i ax_next = _mm256_sign_epi8(w_0_next, w_0_next);
                const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);
                const __m256i ax2_next = _mm256_sign_epi8(w_128_next, w_128_next);

                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                __m256i activation_next = a_start[2];
                __m256i activation2_next = a_start[3];

                const __m256i sy = _mm256_sign_epi8(activation, w_0);
                const __m256i sy_next = _mm256_sign_epi8(activation_next, w_0_next);
                const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);
                const __m256i sy2_next = _mm256_sign_epi8(activation2_next, w_128_next);

                dot = _mm256_maddubs_epi16(ax, sy);
                dot2 = _mm256_maddubs_epi16(ax2, sy2);
                dot3 = _mm256_maddubs_epi16(ax_next, sy_next);
                dot4 = _mm256_maddubs_epi16(ax2_next, sy2_next);

                const __m256i ones = _mm256_set1_epi16(1);
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                const __m256i summed_pairs3 = _mm256_madd_epi16(ones, dot3);
                const __m256i summed_pairs4 = _mm256_madd_epi16(ones, dot4);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);
                __m256 intermediate3 = _mm256_cvtepi32_ps(summed_pairs3);
                __m256 intermediate4 = _mm256_cvtepi32_ps(summed_pairs4);

                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
                __m256 v_s3 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);
                __m256 v_s4 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);
                accumulator = _mm256_fmadd_ps(intermediate, v_s, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate2, v_s2, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate3, v_s3, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate4, v_s4, accumulator);
                s_ptr += 4;
                sa_ptr += 4;
                w_start += 2;
                a_start += 4;
            }
            auto *ptr = (float *)&accumulator;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }

    return nullptr;
}

struct w4a8_thread_args_transposed
{
    int start_i, end_i;
    const struct matmul_params *params;
};


//TODO
static void *all_techniques_worker_func_transposed(void *args)
{
    auto *mat_args = (struct w4a8_thread_args_transposed *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size; // block_size = 32

    for (int row = mat_args->start_i; row < mat_args->end_i; row++)
    {
        for (int col = 0; col < n; col++)
        {
#ifdef QM_ARM
            // ...existing code for ARM SIMD...
#endif
#ifdef QM_x86
            // ...existing code for x86 SIMD...
#endif
        }
    }

    return nullptr;
}

namespace matmul
{
    void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params)
    {
        int i, j, k;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        const int block_size = params->block_size;
        float *scale = params->scales, *offset = params->offset;

        assert(params->block_size % 32 == 0); // support block size to be multiples of 32
        assert(A->row == C->row);             // support block size to be multiples of 32

        quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

        constexpr  int num_thread = 8;
        pthread_t thread_pool[num_thread];
        struct w4a8_thread_args threads_args[num_thread];
        assert(params->block_size == 32); // support block size 32 for now

        for (int j = 0; j < num_thread; j++)
        {
            threads_args[j].params = params;
            threads_args[j].start_j = j * (C->column / num_thread);
            threads_args[j].end_j = (j + 1) * (C->column / num_thread);
            pthread_create(&thread_pool[j], nullptr, all_techniques_worker_func, &threads_args[j]);
        }

        for (unsigned long j : thread_pool)
        {
            pthread_join(j, nullptr);
        }
    }

    void MatmulOperator::mat_mul_transposed_all_techniques(struct matmul_params* params)
    {
        const int num_thread = 8;
        pthread_t thread_pool[num_thread];
        struct w4a8_thread_args_transposed threads_args[num_thread];
        const struct matrix *C = &params->C;

        for (int i = 0; i < num_thread; i++)
        {
            threads_args[i].params = params;
            threads_args[i].start_i = i * (C->row / num_thread);
            threads_args[i].end_i = (i + 1) * (C->row / num_thread);
            pthread_create(&thread_pool[i], nullptr, all_techniques_worker_func_transposed, &threads_args[i]);
        }

        for (unsigned long i : thread_pool)
        {
            pthread_join(i, nullptr);
        }
    }
} // namespace matmul