/*
 * count_explore.cu — Fused hash → count → reward CUDA kernel.
 *
 * One kernel launch computes intrinsic reward for all observations:
 *   1. Hash each observation to a bucket index
 *   2. Atomically increment the count for that bucket
 *   3. Compute reward = beta / sqrt(count)
 *
 * This replaces 3 separate PyTorch operations (hash computation,
 * scatter_add, index lookup + division) with a single fused kernel.
 *
 * Compile: nvcc -O3 -shared -Xcompiler -fPIC count_explore.cu -o count_explore.so
 * Load:    torch.utils.cpp_extension.load(...)
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

/*
 * Fused intrinsic reward kernel for count-based exploration.
 *
 * @param obs           Observations, shape (n, obs_dim), row-major float32
 * @param hash_counts   Global count table, shape (n_buckets,), int32
 * @param rewards       Output intrinsic rewards, shape (n,), float32
 * @param n             Number of observations
 * @param obs_dim       Observation dimensionality
 * @param n_buckets     Hash table size
 * @param beta          Reward scaling factor
 * @param hash_scale    Discretization scale (default 97.0)
 */
__global__ void count_explore_kernel(
    const float* __restrict__ obs,
    int*         __restrict__ hash_counts,
    float*       __restrict__ rewards,
    const int n,
    const int obs_dim,
    const int n_buckets,
    const float beta,
    const float hash_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* Step 1: Hash the observation (sum of discretized values mod n_buckets) */
    int hash_val = 0;
    const float* row = obs + idx * obs_dim;
    for (int i = 0; i < obs_dim; i++) {
        hash_val += (int)(row[i] * hash_scale);
    }
    hash_val = abs(hash_val) % n_buckets;

    /* Step 2: Atomically increment count */
    int count = atomicAdd(&hash_counts[hash_val], 1) + 1;

    /* Step 3: Compute reward = beta / sqrt(count) */
    rewards[idx] = beta / sqrtf((float)count);
}

/*
 * Fused reward combination kernel.
 *
 * Computes: rewards[i] = ext_rewards[i] + beta * clamp(int_rewards[i], 0, clip)
 *
 * Fuses what would be 3 separate PyTorch ops (clamp, multiply, add).
 */
__global__ void combine_rewards_kernel(
    float*       __restrict__ rewards,
    const float* __restrict__ ext_rewards,
    const float* __restrict__ int_rewards,
    const int n,
    const float beta,
    const float clip_min,
    const float clip_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ir = int_rewards[idx];
    ir = fminf(fmaxf(ir, clip_min), clip_max);  /* clamp */
    rewards[idx] = ext_rewards[idx] + beta * ir;
}

/*
 * Running reward normalization kernel.
 *
 * Normalizes intrinsic rewards in-place using provided running mean/var:
 *   rewards[i] = clamp((rewards[i] - mean) / sqrt(var + eps), -5, 5)
 */
__global__ void normalize_rewards_kernel(
    float*       __restrict__ rewards,
    const int n,
    const float mean,
    const float var,
    const float eps,
    const float clip_min,
    const float clip_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float r = rewards[idx];
    r = (r - mean) / sqrtf(var + eps);
    rewards[idx] = fminf(fmaxf(r, clip_min), clip_max);
}

}  /* extern "C" */
