#include <assert.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define COUNT_OF(x) (sizeof(x) / sizeof(x[0]))
#define C(m, x, y) (m->data[(x) * (m->size) + (y)])


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
    matrix_t* m = malloc(sizeof(matrix_t));

    m->data = data;
    m->size = size;

    return m;
}

void matrix_free(matrix_t* m) {
    free(m->data);
    free(m);
}

void matrix_cpy(matrix_t* dst, const matrix_t* src) {
    assert(dst->size == src->size);
    memcpy(dst->data, src->data, src->size * src->size * sizeof(float));
}

void matrix_zero(matrix_t* dst) {
    memset(dst->data, 0, dst->size * dst->size * sizeof(float));
}

void matrix_print(const matrix_t* m) {
    for (uint32_t i = 0; i < m->size; i++) {
        for (uint32_t j = 0; j < m->size; j++) {
            printf("%10.2f", C(m, i, j));
        }
        printf("\n");
    }
}

mblock_t* mblock_alloc(uint32_t size, uint32_t msize) {
    matrix_t** data = (matrix_t**)calloc(size * size, sizeof(matrix_t*));
    mblock_t* b = malloc(sizeof(mblock_t));

    for (uint32_t i = 0; i < size * size; i++) {
        data[i] = matrix_alloc(msize);
    }

    b->data = data;
    b->size = size;

    return b;
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

void mblock_fill(mblock_t* dst, const matrix_t* src) {
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
            matrix_t* submatrix = C(src, i / (src->data[0]->size), j / (src->data[0]->size));
            C(dst, i, j) = C(submatrix, i % (src->data[0]->size), j % (src->data[0]->size));
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
        return ;
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

void matrix_add(const matrix_t* a, const matrix_t* b, matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    for (uint32_t i = 0; i < a->size * a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void matrix_sub(const matrix_t* a, const matrix_t* b, matrix_t* out) {
    assert(a->size == b->size);
    assert(a->size == out->size);

    for (uint32_t i = 0; i < a->size * a->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void matrix_mul(const matrix_t* a, const matrix_t* b, matrix_t* out) {
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

uint32_t matrix_sqrt(const matrix_t* a, matrix_t* output, matrix_t* Y, matrix_t* Z, matrix_t* nextY, matrix_t* nextZ, matrix_t* tmp, float err) {
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

void matrix_transpose(const matrix_t* a, matrix_t* out) {
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
    matrix_sqrt(C(a, n - 1, n - 1), C(l, n - 1, n - 1), Y, Z, nextY, nextZ, tmp1, 0.0001);

    mblock_free(a);
    matrix_free(tmp1);
    matrix_free(tmp2);
    matrix_free(Y);
    matrix_free(Z);
    matrix_free(nextY);
    matrix_free(nextZ);
}

void decomp2_block_mpi(const mblock_t* mblock, mblock_t* l) {
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
        #pragma omp parallel for
        for (uint32_t i = k + 1; i < n; i++) {
            // matrix_div(C(a, i, k), C(l, k, k), C(l, i, k));
            matrix_mul(C(a, i, k), tmp1, C(l, i, k));
        }

        #pragma omp parallel for
        for (uint32_t j = k + 1; j < n; j++) {
            for (uint32_t i = j; i < n; i++) {
                matrix_transpose(C(l, j, k), tmp1);
                matrix_mul(C(l, i, k), tmp1, tmp2);
                matrix_sub(C(a, i, j), tmp2, C(a, i, j));
            }
        }
    }
    matrix_sqrt(C(a, n - 1, n - 1), C(l, n - 1, n - 1), Y, Z, nextY, nextZ, tmp1, 0.0001);

    mblock_free(a);
    matrix_free(tmp1);
    matrix_free(tmp2);
    matrix_free(Y);
    matrix_free(Z);
    matrix_free(nextY);
    matrix_free(nextZ);
}


typedef struct {
    uint32_t start;
    uint32_t end;
    uint32_t k;
    uint32_t n;
    mblock_t* a;
    mblock_t* l;
    matrix_t* inv;
} thread_args;


void* loop1(void* arguments) {
    thread_args* args = (thread_args*)arguments;

    for (uint32_t i = args->start; i < args->end; i++) {
        // matrix_div(C(a, i, k), C(l, k, k), C(l, i, k));
        matrix_mul(C(args->a, i, args->k), args->inv, C(args->l, i, args->k));
    }

    return NULL;
}

void* loop2(void *arguments) {
    thread_args* args = (thread_args*)arguments;

    matrix_t* tmp1 = matrix_alloc(args->a->data[0]->size);
    matrix_t* tmp2 = matrix_alloc(args->a->data[0]->size);
    for (uint32_t j = args->start; j < args->end; j++) {
        for (uint32_t i = j; i < args->n; i++) {
            matrix_transpose(C(args->l, j, args->k), tmp1);
            matrix_mul(C(args->l, i, args->k), tmp1, tmp2);
            matrix_sub(C(args->a, i, j), tmp2, C(args->a, i, j));
        }
    }
    matrix_free(tmp1);
    matrix_free(tmp2);

    return NULL;
}

#define NUM_THREADS 12

void decomp2_block_pthreads(const mblock_t* mblock, mblock_t* l) {
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

    pthread_t threads[NUM_THREADS];
    thread_args args[NUM_THREADS];
    for (uint32_t k = 0; k < n - 1; k++) {
        matrix_sqrt(C(a, k, k), C(l, k, k), Y, Z, nextY, nextZ, tmp1, 0.0001);
        matrix_inv(C(l, k, k), tmp2);

        uint32_t chunk_size = (n - k - 1) / NUM_THREADS;
        for (int t = 0; t < NUM_THREADS; t++) {
            args[t].start = k + 1 + t * chunk_size;
            args[t].end = (t == NUM_THREADS - 1) ? n : args[t].start + chunk_size;
            args[t].k = k;
            args[t].a = a;
            args[t].l = l;
            args[t].inv = tmp2;

            pthread_create(&threads[t], NULL, loop1, (void*)&args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        for (int t = 0; t < NUM_THREADS; t++) {
            args[t].start = k + 1 + t * chunk_size;
            args[t].end = (t == NUM_THREADS - 1) ? n : args[t].start + chunk_size;
            args[t].k = k;
            args[t].a = a;
            args[t].n = n;
            args[t].l = l;

            pthread_create(&threads[t], NULL, loop2, (void*)&args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
    }
    matrix_sqrt(C(a, n - 1, n - 1), C(l, n - 1, n - 1), Y, Z, nextY, nextZ, tmp1, 0.0001);

    mblock_free(a);
    matrix_free(tmp1);
    matrix_free(tmp2);
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

    free(mblock);
    free(m_out);
}

void time_block_decompose_mpi(matrix_t* matrix, uint32_t bs) {
    struct timespec start, finish;
    double elapsed;

    assert(matrix->size % bs == 0);

    mblock_t* mblock = mblock_alloc(matrix->size / bs, bs);
    mblock_t* m_out = mblock_alloc(matrix->size / bs, bs);

    mblock_fill(mblock, matrix);

    clock_gettime(CLOCK_MONOTONIC, &start);
    decomp2_block_mpi(mblock, m_out);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("block_mpi,%u,%u,%f\n", matrix->size, bs, elapsed);

    free(mblock);
    free(m_out);
}


void time_block_decompose_phreads(matrix_t* matrix, uint32_t bs) {
    struct timespec start, finish;
    double elapsed;

    assert(matrix->size % bs == 0);

    mblock_t* mblock = mblock_alloc(matrix->size / bs, bs);
    mblock_t* m_out = mblock_alloc(matrix->size / bs, bs);

    mblock_fill(mblock, matrix);


    clock_gettime(CLOCK_MONOTONIC, &start);
    decomp2_block_pthreads(mblock, m_out);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("block_phreads,%u,%u,%f\n", matrix->size, bs, elapsed);

    free(mblock);
    free(m_out);
}

int32_t main(int argc, char**argv) {
    srand(2137);

    for(uint32_t n = 0; n < 10; n++) {
        for(uint32_t i = 4; i <= 12; i++) {
            matrix_t* matrix = matrix_alloc(1 << i);
            matrix_random_pds(matrix, 50);

            time_block_decompose(matrix, 4);
            time_block_decompose_phreads(matrix, 4);
            time_block_decompose_mpi(matrix, 4);

            matrix_free(matrix);
        }
    }

    return 0;
}


