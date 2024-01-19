#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define COUNT_OF(x) (sizeof(x) / sizeof(x[0]))
#define C(m, x, y) (m->data[(x) * (m->size) + (y)])

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    float* data;
    uint32_t size;
} matrix_t;

typedef struct {
    matrix_t** data;
    uint32_t size;
} mblock_t;

matrix_t* matrix_alloc(uint32_t size) {
    // calloc will memset 0
    float* data = (float*)calloc(size * size, sizeof(float));
    matrix_t* m = (matrix_t*)malloc(sizeof(matrix_t));

    m->data = data;
    m->size = size;

    return m;
}

matrix_t* matrix_alloc_cuda(uint32_t size) {
    float* data;
    gpuErrchk(cudaMallocManaged(&data, size * size * sizeof(float)));
    memset(data, 0, size * size * sizeof(float));

    matrix_t* m;
    gpuErrchk(cudaMallocManaged(&m, sizeof(matrix_t)));

    m->data = data;
    m->size = size;

    return m;
}

void matrix_free(matrix_t* m) {
    free(m->data);
    free(m);
}

void matrix_free_cuda(matrix_t* m) {
    cudaFree(m->data);
    cudaFree(m);
}

void matrix_cpy(matrix_t* dst, const matrix_t* src) {
    assert(dst->size == src->size);
    memcpy(dst->data, src->data, src->size * src->size * sizeof(float));
}

__host__ __device__ void matrix_zero(matrix_t* dst) {
    memset(dst->data, 0, dst->size * dst->size * sizeof(float));
}

__host__ __device__ void matrix_print(const matrix_t* m) {
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = 0; j < m->size; j++) {
            printf("%10.2f", C(m, i, j));
        }
        printf("\n");
    }
}

mblock_t* mblock_alloc(uint32_t size, uint32_t msize) {
    matrix_t** data = (matrix_t**)calloc(size * size, sizeof(matrix_t*));
    mblock_t* b = (mblock_t*)malloc(sizeof(mblock_t));

    for (uint32_t i = 0; i < size * size; i++) {
        data[i] = matrix_alloc(msize);
    }

    b->data = data;
    b->size = size;

    return b;
}

mblock_t* mblock_alloc_cuda(uint32_t size, uint32_t msize) {
    matrix_t** data;
    gpuErrchk(cudaMallocManaged(&data, size * size * sizeof(matrix_t*)));
    mblock_t* b;
    gpuErrchk(cudaMallocManaged(&b, sizeof(mblock_t)));

    for (uint32_t i = 0; i < size * size; i++) {
        data[i] = matrix_alloc_cuda(msize);
    }

    b->data = data;
    b->size = size;

    return b;
}

void mblock_free_cuda(mblock_t* b) {
    for (uint32_t i = 0; i < b->size * b->size; i++) {
        matrix_free_cuda(b->data[i]);
    }

    cudaFree(b->data);
    cudaFree(b);
}

void mblock_free(mblock_t* b) {
    for (uint32_t i = 0; i < b->size * b->size; i++) {
        matrix_free(b->data[i]);
    }

    free(b->data);
    free(b);
}

void mblock_cpy(mblock_t* dst, const mblock_t* src) {
    for (uint32_t i = 0; i < src->size; i++) {
        for (uint32_t j = 0; j < src->size; j++) {
            matrix_cpy(C(dst, i, j), C(src, i, j));
        }
    }
}

void mblock_zero(mblock_t* dst) {
    for (uint32_t i = 0; i < dst->size; i++) {
        for (uint32_t j = 0; j < dst->size; j++) {
            matrix_zero(C(dst, i, j));
        }
    }
}

void mblock_print(const mblock_t* m) {
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = 0; j < m->size; j++) {
            printf("== x: %d, y: %d ==\n", i, j);
            matrix_print(C(m, i, j));
            printf("==================\n");
        }
        printf("\n");
    }
}

__host__ __device__ void mblock_fill(mblock_t* dst, const matrix_t* src) {
    assert(src->size % dst->size == 0);
    for (uint32_t i = 0; i < src->size; i++) {
        for (uint32_t j = 0; j < src->size; j++) {
            matrix_t* m = C(dst, i / (dst->data[0]->size), j / (dst->data[0]->size));
            C(m, i % (dst->data[0]->size), j % (dst->data[0]->size)) = C(src, i, j);
        }
    }
}

void matrix_reconstruct(matrix_t* dst, const mblock_t* src) {
    assert(dst->size % src->size == 0);

    for (uint32_t i = 0; i < dst->size; i++) {
        for (uint32_t j = 0; j < dst->size; j++) {
            matrix_t* submatrix =
                C(src, i / (src->data[0]->size), j / (src->data[0]->size));
            C(dst, i, j) = C(submatrix, i % (src->data[0]->size),
                             j % (src->data[0]->size));
        }
    }
}

float det(const matrix_t* m) {
    if (m->size == 1)
        return m->data[0];
    if (m->size == 2)
        return m->data[0] * m->data[3] - m->data[1] * m->data[2];

    matrix_t* minor = matrix_alloc(m->size - 1);
    float d = 0;
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = 1; j < m->size; j++) {
            for (uint32_t k = 0; k < m->size; k++) {
                if (k < i)
                    C(minor, j - 1, k) = C(m, j, k);
                else if (k > i)
                    C(minor, j - 1, k - 1) = C(m, j, k);
            }
        }
        d += ((i % 2 == 0) ? 1 : -1) * m->data[i] * det(minor);
    }
    matrix_free(minor);

    return d;
}

void decomp2(const matrix_t* matrix, matrix_t* l) {
    matrix_t* a = matrix_alloc(matrix->size);

    matrix_cpy(a, matrix);
    matrix_zero(l);

    for (uint32_t k = 0; k < matrix->size - 1; k++) {
        C(l, k, k) = sqrt(C(a, k, k));
        for (uint32_t i = k + 1; i < matrix->size; i++) {
            C(l, i, k) = C(a, i, k) / C(l, k, k);
        }
        for (uint32_t j = k + 1; j < matrix->size; j++) {
            for (uint32_t i = j; i < matrix->size; i++) {
                C(a, i, j) -= C(l, i, k) * C(l, j, k);
            }
        }
    }
    // last element
    l->data[l->size * l->size - 1] = sqrt(a->data[l->size * l->size - 1]);

    matrix_free(a);
}

float cof(const matrix_t* m, matrix_t* minor, uint32_t row, uint32_t col) {
    uint32_t minorRow = 0, minorCol = 0;

    for (uint32_t i = 0; i < m->size; i++) {
        if (i == row)
            continue;
        minorCol = 0;
        for (uint32_t j = 0; j < m->size; j++) {
            if (j == col)
                continue;
            C(minor, minorRow, minorCol) = C(m, i, j);
            minorCol++;
        }
        minorRow++;
    }

    float c = ((row + col) % 2 == 0 ? 1 : -1) * det(minor);

    return c;
}

void matrix_inv(const matrix_t* m, matrix_t* out) {
    assert(m->size == out->size);

    float d = det(m);
    if (m->size == 1) {
        out->data[0] = 1 / d;
        return;
    }
    if (d == 0) {
        for (uint32_t i = 0; i < m->size * m->size; i++) {
            out->data[i] = 0;
        }
        return;
    }

    matrix_t* minor = matrix_alloc(m->size - 1);
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = 0; j < m->size; j++) {
            C(out, j, i) = cof(m, minor, i, j) / d;
        }
    }
    matrix_free(minor);
}

__host__ __device__ void matrix_add(const matrix_t* a, const matrix_t* b,
                                    matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    for (uint32_t i = 0; i < a->size * a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

__host__ __device__ void matrix_sub(const matrix_t* a, const matrix_t* b,
                                    matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    for (uint32_t i = 0; i < a->size * a->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

__host__ __device__ void matrix_mul(const matrix_t* a, const matrix_t* b,
                                    matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    matrix_zero(out);

    for (uint32_t i = 0; i < a->size; i++) {
        for (uint32_t j = 0; j < a->size; j++) {
            for (uint32_t k = 0; k < a->size; k++) {
                C(out, i, j) += C(a, i, k) * C(b, k, j);
            }
        }
    }
}

void matrix_div(const matrix_t* a, const matrix_t* b, matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    matrix_t* inverseB = matrix_alloc(a->size);
    matrix_inv(b, inverseB);
    matrix_mul(a, inverseB, out);
    matrix_free(inverseB);
}

void matrix_mul_imm(const matrix_t* a, float imm, matrix_t* out) {
    assert(a->size == out->size);

    for (uint32_t i = 0; i < a->size * a->size; i++) {
        out->data[i] = a->data[i] * imm;
    }
}

float matrix_err(const matrix_t* a, const matrix_t* b) {
    assert(a->size == b->size);

    float norm = 0.0f;
    for (uint32_t i = 0; i < a->size * a->size; i++) {
        float diff = a->data[i] - b->data[i];
        norm += diff * diff;
    }

    return sqrt(norm);
}

void matrix_i(matrix_t* output) {
    for (uint32_t i = 0; i < output->size; ++i) {
        for (uint32_t j = 0; j < output->size; ++j) {
            C(output, i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }
}

uint32_t matrix_sqrt(const matrix_t* a, matrix_t* output, matrix_t* Y,
                     matrix_t* Z, matrix_t* nextY, matrix_t* nextZ,
                     matrix_t* tmp, float err) {
    matrix_cpy(Y, a);
    matrix_i(Z);

    uint32_t i = 0;
    uint8_t done = 0;

    if (a->size == 1) {
        C(output, 0, 0) = sqrtf(C(a, 0, 0));
        return 0;
    }
    while (!done) {
        matrix_inv(Z, tmp);
        matrix_add(Y, tmp, tmp);
        matrix_mul_imm(tmp, 0.5, nextY);

        matrix_inv(Y, tmp);
        matrix_add(Z, tmp, tmp);
        matrix_mul_imm(tmp, 0.5, nextZ);

        done = (matrix_err(Y, nextY) < err && matrix_err(Z, nextZ) < err);

        matrix_cpy(Y, nextY);
        matrix_cpy(Z, nextZ);
        i++;
    }

    matrix_cpy(output, Y);

    return i;
}

__device__ __host__ void matrix_transpose(const matrix_t* a, matrix_t* out) {
    for (uint32_t i = 0; i < a->size; i++) {
        for (uint32_t j = 0; j < a->size; j++) {
            C(out, i, j) = C(a, j, i);
        }
    }
}

void matrix_random_l(matrix_t* m, uint64_t max_val) {
    uint32_t min_val = 1;
    for (uint32_t i = 0; i < m->size; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            if (min_val == max_val) {
                C(m, i, j) = min_val;
            } else {
                C(m, i, j) = rand() % (max_val - min_val + 1) + min_val;
            }
        }
    }
}

void matrix_random_pds1(matrix_t* m, int64_t min_val) {
    matrix_t* tmp = matrix_alloc(m->size);
    matrix_t* tmp2 = matrix_alloc(m->size);

    matrix_random_l(m, min_val);
    matrix_transpose(m, tmp);
    matrix_mul(m, tmp, tmp2);
    matrix_cpy(m, tmp2);

    matrix_free(tmp);
    matrix_free(tmp2);
}

uint32_t matrix_cmp(const matrix_t* m1, const matrix_t* m2, float perr) {
    uint32_t c = 0;
    if (m1->size != m2->size) {
        return UINT32_MAX;
    }

    for (uint32_t i = 0; i < m1->size; ++i) {
        for (uint32_t j = 0; j < m1->size; ++j) {
            float diff = fabsf(C(m1, i, j) - C(m2, i, j));
            float allowed_err = perr * fabsf(C(m1, i, j));
            if (diff > allowed_err) {
                c++;
            }
        }
    }

    return c;
}

void decomp2_block(const mblock_t* mblock, mblock_t* l) {
    mblock_t* a = mblock_alloc(mblock->size, mblock->data[0]->size);
    matrix_t* tmp1 = matrix_alloc(mblock->data[0]->size);
    matrix_t* tmp2 = matrix_alloc(mblock->data[0]->size);
    // optimization reasons
    matrix_t* Y = matrix_alloc(tmp1->size);
    matrix_t* Z = matrix_alloc(tmp1->size);
    matrix_t* nextY = matrix_alloc(tmp1->size);
    matrix_t* nextZ = matrix_alloc(tmp1->size);

    uint32_t n = mblock->size;

    mblock_cpy(a, mblock);
    mblock_zero(l);

    for (uint32_t k = 0; k < n - 1; k++) {
        matrix_sqrt(C(a, k, k), C(l, k, k), Y, Z, nextY, nextZ, tmp1, 0.0001);
        matrix_inv(C(l, k, k), tmp1);

        for (uint32_t i = k + 1; i < n; i++) {
            // matrix_div(C(a, i, k), C(l, k, k), C(l, i, k));
            matrix_mul(C(a, i, k), tmp1, C(l, i, k));
        }

        for (uint32_t j = k + 1; j < n; j++) {
            for (uint32_t i = j; i < n; i++) {
                matrix_transpose(C(l, j, k), tmp1);
                matrix_mul(C(l, i, k), tmp1, tmp2);
                matrix_sub(C(a, i, j), tmp2, C(a, i, j));
            }
        }
    }
    matrix_sqrt(C(a, n - 1, n - 1), C(l, n - 1, n - 1), Y, Z, nextY, nextZ,
                tmp1, 0.0001);

    mblock_free(a);
    matrix_free(tmp1);
    matrix_free(tmp2);
    matrix_free(Y);
    matrix_free(Z);
    matrix_free(nextY);
    matrix_free(nextZ);
}

__global__ void loop1(uint32_t n, uint32_t k, mblock_t* l, const mblock_t* a,
                      const matrix_t* tmp) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (uint32_t i = k + index + 1; i < n; i += stride) {
        // matrix_div(C(a, i, k), C(l, k, k), C(l, i, k));
        matrix_mul(C(a, i, k), tmp, C(l, i, k));
    }
}

__global__ void loop2(uint32_t n, uint32_t k, mblock_t* l, mblock_t* a) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    assert(a->data[0]->size == 4);

    float tmp1_data[16];
    float tmp2_data[16];
    matrix_t tmp1 = {.data = tmp1_data, .size = 4};
    matrix_t tmp2 = {.data = tmp2_data, .size = 4};

    for (uint32_t j = k + index + 1; j < n; j += stride) {
        for (uint32_t i = j; i < n; i++) {
            matrix_transpose(C(l, j, k), &tmp1);
            matrix_mul(C(l, i, k), &tmp1, &tmp2);
            matrix_sub(C(a, i, j), &tmp2, C(a, i, j));
        }
    }
}

void decomp2_block_pararell(const mblock_t* mblock, mblock_t* l) {
    mblock_t* a = mblock_alloc_cuda(mblock->size, mblock->data[0]->size);
    matrix_t* tmp1 = matrix_alloc_cuda(mblock->data[0]->size);
    // optimization reasons
    matrix_t* Y = matrix_alloc(tmp1->size);
    matrix_t* Z = matrix_alloc(tmp1->size);
    matrix_t* nextY = matrix_alloc(tmp1->size);
    matrix_t* nextZ = matrix_alloc(tmp1->size);

    uint32_t n = mblock->size;

    mblock_cpy(a, mblock);
    mblock_zero(l);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (uint32_t k = 0; k < n - 1; k++) {
        matrix_sqrt(C(a, k, k), C(l, k, k), Y, Z, nextY, nextZ, tmp1, 0.0001);
        matrix_inv(C(l, k, k), tmp1);

        loop1<<<numBlocks, blockSize>>>(n, k, l, a, tmp1);
        cudaDeviceSynchronize();

        loop2<<<numBlocks, blockSize>>>(n, k, l, a);
        cudaDeviceSynchronize();
    }
    matrix_sqrt(C(a, n - 1, n - 1), C(l, n - 1, n - 1), Y, Z, nextY, nextZ,
                tmp1, 0.0001);

    mblock_free_cuda(a);
    matrix_free_cuda(tmp1);
    matrix_free(Y);
    matrix_free(Z);
    matrix_free(nextY);
    matrix_free(nextZ);
}

void decomp1(const float* matrix, float* lower, uint32_t n) {
    memset(lower, 0, n * n * sizeof(float));

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float sum = 0;

            if (j == i) {
                for (uint32_t k = 0; k < j; k++) {
                    sum += pow(lower[j * n + k], 2);
                }
                lower[j * n + j] = sqrt(matrix[j * n + j] - sum);
            } else {
                for (uint32_t k = 0; k < j; k++) {
                    sum += (lower[i * n + k] * lower[j * n + k]);
                }
                lower[i * n + j] = (matrix[i * n + j] - sum) / lower[j * n + j];
            }
        }
    }
}

void matrix_random_pds(matrix_t* m, int64_t max_val) {
    int64_t min_val = 1;
    matrix_t* tmp1 = matrix_alloc(m->size);
    matrix_t* tmp2 = matrix_alloc(m->size);
    // Fill the matrix with random values in the specified range
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = i; j < m->size; j++) {
            uint64_t random_value = rand() % (max_val - min_val) + min_val;
            C(m, i, j) = random_value;
        }
    }
    matrix_transpose(m, tmp1);
    matrix_add(m, tmp1, tmp2);
    matrix_mul_imm(tmp2, 0.5, m);

    // Add N * Identity matrix
    for (uint32_t i = 0; i < m->size; i++) {
        C(m, i, i) += m->size * 8;
    }
}

void time_normal_decompose(matrix_t* matrix) {
    struct timespec start, finish;
    double elapsed;

    matrix_t* m_out = matrix_alloc(matrix->size);

    clock_gettime(CLOCK_MONOTONIC, &start);
    decomp2(matrix, m_out);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("standard,%u,0,%f\n", matrix->size, elapsed);

    free(m_out);
}

void time_block_decompose(matrix_t* matrix, uint32_t bs) {
    struct timespec start, finish;
    double elapsed;

    assert(matrix->size % bs == 0);

    mblock_t* mblock = mblock_alloc(matrix->size / bs, bs);
    mblock_t* m_out = mblock_alloc(matrix->size / bs, bs);

    mblock_fill(mblock, matrix);

    clock_gettime(CLOCK_MONOTONIC, &start);
    decomp2_block(mblock, m_out);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("block,%u,%u,%f\n", matrix->size, bs, elapsed);

    matrix_t* out = matrix_alloc(matrix->size);
    matrix_t* outT = matrix_alloc(matrix->size);
    matrix_t* res = matrix_alloc(matrix->size);

    free(mblock);
    free(m_out);
}

void time_block_decompose_pararell(matrix_t* matrix, uint32_t bs) {
    struct timespec start, finish;
    double elapsed;

    assert(matrix->size % bs == 0);

    mblock_t* mblock = mblock_alloc_cuda(matrix->size / bs, bs);
    mblock_t* m_out = mblock_alloc_cuda(matrix->size / bs, bs);

    mblock_fill(mblock, matrix);

    clock_gettime(CLOCK_MONOTONIC, &start);
    decomp2_block_pararell(mblock, m_out);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("block_pararell_cuda,%u,%u,%f\n", matrix->size, bs, elapsed);

    mblock_free_cuda(mblock);
    mblock_free_cuda(m_out);
}

int32_t main(int argc, char** argv) {
    srand(2137);

    for (uint32_t n = 0; n < 10; n++) {
        for (uint32_t i = 4; i <= 11; i++) {
            matrix_t* matrix = matrix_alloc(1 << i);
            matrix_random_pds(matrix, 50);

            time_block_decompose(matrix, 4);
            time_block_decompose_pararell(matrix, 4);

            matrix_free(matrix);
        }
    }

    return 0;
}
