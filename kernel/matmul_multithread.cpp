#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "common.h"
#include "operators.h"

#include "threadPool.h" // 引入我们的线程池
#include "common.h"     // 为了 NUM_THREAD_MATMUL

static threadPool g_thread_pool(NUM_THREAD_MATMUL);

struct multithreading_thread_args {
    int start, end;
    const struct matmul_params* params;
};
static void* multithreading_worker_func(void* args) {
    auto* mat_args = (struct multithreading_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col++) {
            float acc = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t* w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation
                const signed char* a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
#ifdef QM_ARM
                // order of weights with QM_ARM:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
                // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         128 bit                         127
                // process 16 bytes of weigths (128 bit) = 1 block
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0;
                // process 16 bytes of weigths (128 bit)
                for (int qj = 0; qj < 16; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum += a_int8[qj + 16] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    auto w_de_0 =static_cast<signed char>(packed_int4_0 & 0x0F)-8;
                    auto w_de_16 =static_cast<signed char>(packed_int4_0 >> 4) - 8;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += static_cast<float>(intermediate_sum) * s_a * s_w;
                acc += static_cast<float>(intermediate_sum_2nd) * s_a_2nd * s_w_2nd;
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc;
        }
    }
    return NULL;
}

namespace matmul {
    void MatmulOperator::mat_mul_multithreading(struct matmul_params* params) {
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        const int block_size = params->block_size;

        quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

        int n = C->column;
        const int num_thread = NUM_THREAD_MATMUL;

        // 使用 std::vector 来存储线程参数，以确保其生命周期在任务执行时仍然有效
        std::vector<multithreading_thread_args> threads_args(num_thread);

        // 提交任务到全局线程池
        for (int j = 0; j < num_thread; j++) {
            threads_args[j].params = params;
            threads_args[j].start = j * (n / num_thread);
            threads_args[j].end = (j + 1) * (n / num_thread);

            // 捕获指向当前参数的指针，并提交给线程池
            g_thread_pool.submit([args = &threads_args[j]]() {
                multithreading_worker_func(args);
            });
        }

        // 等待线程池完成这一批次的所有任务
        g_thread_pool.wait_for_completion();
    };

}  // namespace matmul