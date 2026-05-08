#include "utils.h"
#include "gpu_kernels.h"
#include "reduce_vector_sum.h"
#include "aux_funs.h"

#define MAX_DIMS 6
#define N_THREADS 256
//#define FULL_LEDOIT_WOLF_ESTIMATOR

__constant__ static float c_mean[MAX_DIMS];
__constant__ static float c_covar_inv[(MAX_DIMS*MAX_DIMS + MAX_DIMS) / 2];
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
__constant__ static float c_covar[(MAX_DIMS*MAX_DIMS + MAX_DIMS) / 2];
#endif
//__constant__ static float c_A; // gaussian normalize constant

// Reuse device buffers across calls to avoid frequent cudaMalloc/cudaFree overhead.
static float* s_d_space = nullptr;
static float* s_d_partial_stats = nullptr;
static float* s_d_reduced_stats = nullptr;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static float* s_d_weight = nullptr;
static float* s_d_weighted_covar = nullptr;
static float* s_d_weighted_LW_b2 = nullptr;
#endif

static size_t s_space_capacity_bytes = 0;
static size_t s_partial_stats_capacity_bytes = 0;
static size_t s_reduced_stats_capacity_bytes = 0;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static size_t s_weight_capacity_bytes = 0;
static size_t s_weighted_covar_capacity_bytes = 0;
static size_t s_weighted_LW_b2_capacity_bytes = 0;
#endif

static float* s_ht_weight = nullptr;
static float* s_ht_mean = nullptr;
static float* s_ht_covar = nullptr;
static float* s_ht_covar_inv = nullptr;
static double* s_ht_covar_full = nullptr;
static double* s_ht_covar_inv_full = nullptr;
static float* s_ht_stats = nullptr;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static float* s_ht_LW_b2 = nullptr;
#endif

static size_t s_ht_weight_capacity = 0;
static size_t s_ht_mean_capacity = 0;
static size_t s_ht_covar_capacity = 0;
static size_t s_ht_covar_inv_capacity = 0;
static size_t s_ht_covar_full_capacity = 0;
static size_t s_ht_covar_inv_full_capacity = 0;
static size_t s_ht_stats_capacity = 0;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static size_t s_ht_LW_b2_capacity = 0;
#endif

#define MAX_COVAR_DIMS ((MAX_DIMS * MAX_DIMS + MAX_DIMS) / 2)
#define MAX_STATS_DIMS (1 + MAX_DIMS + MAX_COVAR_DIMS)

template <typename T>
static void ensure_device_buffer_capacity(T*& d_ptr, size_t& capacity_bytes, size_t required_bytes) {
	if (required_bytes <= capacity_bytes)
		return;
	if (d_ptr != nullptr) {
		cudaFree(d_ptr);
		d_ptr = nullptr;
		capacity_bytes = 0;
	}
	cudaMalloc((void**)&d_ptr, required_bytes);
	capacity_bytes = required_bytes;
}

template <typename T>
static void ensure_host_buffer_capacity(T*& h_ptr, size_t& capacity, size_t required_count) {
	if (required_count <= capacity)
		return;
	if (h_ptr != nullptr) {
		delete[] h_ptr;
		h_ptr = nullptr;
		capacity = 0;
	}
	h_ptr = new T[required_count];
	capacity = required_count;
}

template <typename T1, typename T2>
static void covar_half_to_full(const T1* half, T2* full, const int dims) {
	for (int d1 = 0; d1 < dims; d1++) {
		for (int d2 = 0; d2 <= d1; d2++) {
			full[d1*dims + d2] = (T2)half[(d1*d1 + d1) / 2 + d2];
			if (d1 != d2)
				full[d2*dims + d1] = full[d1*dims + d2];
		}
	}
}
template <typename T1, typename T2>
static void covar_full_to_half(const T1* full, T2* half, const int dims) {
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			half[(d1*d1 + d1) / 2 + d2] = (T1)full[d1*dims + d2];
}

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
__global__ static void compute_LW_b2(float* d_weight, float* d_weighted_covar, float* d_o_weighted_LW_b2, int N, int dims) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		const int covar_dims = (dims*dims + dims) / 2;
		float weight = d_weight[idx];
		float norm2 = 0;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++) {
				float tmp = d_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] - weight * c_covar[(d1*d1 + d1) / 2 + d2];
				if (d1 != d2)
					norm2 += 2 * tmp*tmp;
				else
					norm2 += tmp * tmp;
			}
		d_o_weighted_LW_b2[idx] = norm2;
	}
}
#endif

// Legacy per-sample e-step is kept out of the build; fused partial reduction below is the active path.
#if 0
// computes weights/weighted-mean/weighted-covar for each sample
__global__ static void e_step( // c_mean, c_covar_inv are needed
	float* d_space,
	float* d_o_weight, float* d_o_weighted_space, float* d_o_weighted_covar,
	float trunc_sigma,
	int N, int dims, float space_scale) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		float diff[MAX_DIMS];

		float scaled_space[MAX_DIMS];
		for (int d = 0; d < dims; d++) {
			scaled_space[d] = d_space[idx*dims + d] * space_scale;
			diff[d] = scaled_space[d] - c_mean[d];
		}


		float z_score = 0;
		for (int d1 = 0; d1 < dims; d1++) {
			float tmp = 0;
			for (int d2 = 0; d2 < dims; d2++) {
				if (d1 >= d2)
					tmp += c_covar_inv[(d1*d1 + d1) / 2 + d2] * diff[d2];
				else
					tmp += c_covar_inv[(d2*d2 + d2) / 2 + d1] * diff[d2];
			}
			z_score += tmp * diff[d1];
		}
		z_score = sqrtf(z_score);

		const float weight = z_score < trunc_sigma ? 1 : 0;

		d_o_weight[idx] = weight;

		for (int d = 0; d < dims; d++)
			d_o_weighted_space[idx*dims + d] = weight * scaled_space[d];

		const int covar_dims = (dims*dims + dims) / 2;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				d_o_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] = weight * diff[d1] * diff[d2];

	}
}

__global__ static void e_step_6(
	float* d_space,
	float* d_o_weight, float* d_o_weighted_space, float* d_o_weighted_covar,
	float trunc_sigma,
	int N, float space_scale) {

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= N)
		return;

	const int base = idx * 6;
	const float s0 = d_space[base + 0] * space_scale;
	const float s1 = d_space[base + 1] * space_scale;
	const float s2 = d_space[base + 2] * space_scale;
	const float s3 = d_space[base + 3] * space_scale;
	const float s4 = d_space[base + 4] * space_scale;
	const float s5 = d_space[base + 5] * space_scale;

	const float diff0 = s0 - c_mean[0];
	const float diff1 = s1 - c_mean[1];
	const float diff2 = s2 - c_mean[2];
	const float diff3 = s3 - c_mean[3];
	const float diff4 = s4 - c_mean[4];
	const float diff5 = s5 - c_mean[5];

	float z_score = 0;
	float tmp = 0;
	tmp += c_covar_inv[0] * diff0;
	tmp += c_covar_inv[1] * diff1;
	tmp += c_covar_inv[3] * diff2;
	tmp += c_covar_inv[6] * diff3;
	tmp += c_covar_inv[10] * diff4;
	tmp += c_covar_inv[15] * diff5;
	z_score += tmp * diff0;

	tmp = 0;
	tmp += c_covar_inv[1] * diff0;
	tmp += c_covar_inv[2] * diff1;
	tmp += c_covar_inv[4] * diff2;
	tmp += c_covar_inv[7] * diff3;
	tmp += c_covar_inv[11] * diff4;
	tmp += c_covar_inv[16] * diff5;
	z_score += tmp * diff1;

	tmp = 0;
	tmp += c_covar_inv[3] * diff0;
	tmp += c_covar_inv[4] * diff1;
	tmp += c_covar_inv[5] * diff2;
	tmp += c_covar_inv[8] * diff3;
	tmp += c_covar_inv[12] * diff4;
	tmp += c_covar_inv[17] * diff5;
	z_score += tmp * diff2;

	tmp = 0;
	tmp += c_covar_inv[6] * diff0;
	tmp += c_covar_inv[7] * diff1;
	tmp += c_covar_inv[8] * diff2;
	tmp += c_covar_inv[9] * diff3;
	tmp += c_covar_inv[13] * diff4;
	tmp += c_covar_inv[18] * diff5;
	z_score += tmp * diff3;

	tmp = 0;
	tmp += c_covar_inv[10] * diff0;
	tmp += c_covar_inv[11] * diff1;
	tmp += c_covar_inv[12] * diff2;
	tmp += c_covar_inv[13] * diff3;
	tmp += c_covar_inv[14] * diff4;
	tmp += c_covar_inv[19] * diff5;
	z_score += tmp * diff4;

	tmp = 0;
	tmp += c_covar_inv[15] * diff0;
	tmp += c_covar_inv[16] * diff1;
	tmp += c_covar_inv[17] * diff2;
	tmp += c_covar_inv[18] * diff3;
	tmp += c_covar_inv[19] * diff4;
	tmp += c_covar_inv[20] * diff5;
	z_score += tmp * diff5;

	z_score = sqrtf(z_score);
	const float weight = z_score < trunc_sigma ? 1 : 0;

	d_o_weight[idx] = weight;

	d_o_weighted_space[base + 0] = weight * s0;
	d_o_weighted_space[base + 1] = weight * s1;
	d_o_weighted_space[base + 2] = weight * s2;
	d_o_weighted_space[base + 3] = weight * s3;
	d_o_weighted_space[base + 4] = weight * s4;
	d_o_weighted_space[base + 5] = weight * s5;

	const int covar_base = idx * 21;
	d_o_weighted_covar[covar_base + 0] = weight * diff0 * diff0;
	d_o_weighted_covar[covar_base + 1] = weight * diff1 * diff0;
	d_o_weighted_covar[covar_base + 2] = weight * diff1 * diff1;
	d_o_weighted_covar[covar_base + 3] = weight * diff2 * diff0;
	d_o_weighted_covar[covar_base + 4] = weight * diff2 * diff1;
	d_o_weighted_covar[covar_base + 5] = weight * diff2 * diff2;
	d_o_weighted_covar[covar_base + 6] = weight * diff3 * diff0;
	d_o_weighted_covar[covar_base + 7] = weight * diff3 * diff1;
	d_o_weighted_covar[covar_base + 8] = weight * diff3 * diff2;
	d_o_weighted_covar[covar_base + 9] = weight * diff3 * diff3;
	d_o_weighted_covar[covar_base + 10] = weight * diff4 * diff0;
	d_o_weighted_covar[covar_base + 11] = weight * diff4 * diff1;
	d_o_weighted_covar[covar_base + 12] = weight * diff4 * diff2;
	d_o_weighted_covar[covar_base + 13] = weight * diff4 * diff3;
	d_o_weighted_covar[covar_base + 14] = weight * diff4 * diff4;
	d_o_weighted_covar[covar_base + 15] = weight * diff5 * diff0;
	d_o_weighted_covar[covar_base + 16] = weight * diff5 * diff1;
	d_o_weighted_covar[covar_base + 17] = weight * diff5 * diff2;
	d_o_weighted_covar[covar_base + 18] = weight * diff5 * diff3;
	d_o_weighted_covar[covar_base + 19] = weight * diff5 * diff4;
	d_o_weighted_covar[covar_base + 20] = weight * diff5 * diff5;
}
#endif

__device__ __forceinline__ static float rg_warp_reduce_sum(float val) {
	for (int offset = 16; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xffffffff, val, offset);
	return val;
}

__global__ static void e_step_partial_reduce(
	float* d_space,
	float* d_o_weight, float* d_o_weighted_covar, float* d_o_partial_stats,
	float trunc_sigma,
	int N, int dims, float space_scale) {

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp_id = tid >> 5;
	const int n_warps = blockDim.x >> 5;
	const int covar_dims = (dims*dims + dims) / 2;
	const int stats_dims = 1 + dims + covar_dims;
	__shared__ float s_warp_partials[8 * MAX_STATS_DIMS];

	float stats[MAX_STATS_DIMS]{ 0 };
	float diff[MAX_DIMS]{ 0 };
	float scaled_space[MAX_DIMS]{ 0 };
	float weight = 0.0f;

	if (idx < N) {
		for (int d = 0; d < dims; d++) {
			scaled_space[d] = d_space[idx*dims + d] * space_scale;
			diff[d] = scaled_space[d] - c_mean[d];
		}

		float z_score = 0;
		for (int d1 = 0; d1 < dims; d1++) {
			float tmp = 0;
			for (int d2 = 0; d2 < dims; d2++) {
				if (d1 >= d2)
					tmp += c_covar_inv[(d1*d1 + d1) / 2 + d2] * diff[d2];
				else
					tmp += c_covar_inv[(d2*d2 + d2) / 2 + d1] * diff[d2];
			}
			z_score += tmp * diff[d1];
		}
		z_score = sqrtf(z_score);
		weight = z_score < trunc_sigma ? 1 : 0;

	#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
		d_o_weight[idx] = weight;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				d_o_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] = weight * diff[d1] * diff[d2];
	#endif
	}

	stats[0] = weight;
	for (int d = 0; d < dims; d++)
		stats[1 + d] = weight * scaled_space[d];
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			stats[1 + dims + (d1*d1 + d1) / 2 + d2] = weight * diff[d1] * diff[d2];

	for (int c = 0; c < stats_dims; c++)
		stats[c] = rg_warp_reduce_sum(stats[c]);

	if (lane == 0) {
		for (int c = 0; c < stats_dims; c++)
			s_warp_partials[warp_id * MAX_STATS_DIMS + c] = stats[c];
	}
	__syncthreads();

	if (warp_id == 0) {
		for (int c = 0; c < stats_dims; c++) {
			float block_val = (lane < n_warps) ? s_warp_partials[lane * MAX_STATS_DIMS + c] : 0.0f;
			block_val = rg_warp_reduce_sum(block_val);
			if (lane == 0)
				d_o_partial_stats[blockIdx.x * MAX_STATS_DIMS + c] = block_val;
		}
	}
}

__global__ static void e_step_6_partial_reduce(
	float* d_space,
	float* d_o_weight, float* d_o_weighted_covar, float* d_o_partial_stats,
	float trunc_sigma,
	int N, float space_scale) {

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp_id = tid >> 5;
	const int n_warps = blockDim.x >> 5;
	__shared__ float s_warp_partials[8 * MAX_STATS_DIMS];

	float stats[MAX_STATS_DIMS]{ 0 };
	float weight = 0.0f;
	float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f;
	float diff0 = 0.0f, diff1 = 0.0f, diff2 = 0.0f, diff3 = 0.0f, diff4 = 0.0f, diff5 = 0.0f;

	if (idx < N) {
		const int base = idx * 6;
		s0 = d_space[base + 0] * space_scale;
		s1 = d_space[base + 1] * space_scale;
		s2 = d_space[base + 2] * space_scale;
		s3 = d_space[base + 3] * space_scale;
		s4 = d_space[base + 4] * space_scale;
		s5 = d_space[base + 5] * space_scale;

		diff0 = s0 - c_mean[0];
		diff1 = s1 - c_mean[1];
		diff2 = s2 - c_mean[2];
		diff3 = s3 - c_mean[3];
		diff4 = s4 - c_mean[4];
		diff5 = s5 - c_mean[5];

		float z_score = 0;
		float tmp = 0;
		tmp += c_covar_inv[0] * diff0;
		tmp += c_covar_inv[1] * diff1;
		tmp += c_covar_inv[3] * diff2;
		tmp += c_covar_inv[6] * diff3;
		tmp += c_covar_inv[10] * diff4;
		tmp += c_covar_inv[15] * diff5;
		z_score += tmp * diff0;

		tmp = 0;
		tmp += c_covar_inv[1] * diff0;
		tmp += c_covar_inv[2] * diff1;
		tmp += c_covar_inv[4] * diff2;
		tmp += c_covar_inv[7] * diff3;
		tmp += c_covar_inv[11] * diff4;
		tmp += c_covar_inv[16] * diff5;
		z_score += tmp * diff1;

		tmp = 0;
		tmp += c_covar_inv[3] * diff0;
		tmp += c_covar_inv[4] * diff1;
		tmp += c_covar_inv[5] * diff2;
		tmp += c_covar_inv[8] * diff3;
		tmp += c_covar_inv[12] * diff4;
		tmp += c_covar_inv[17] * diff5;
		z_score += tmp * diff2;

		tmp = 0;
		tmp += c_covar_inv[6] * diff0;
		tmp += c_covar_inv[7] * diff1;
		tmp += c_covar_inv[8] * diff2;
		tmp += c_covar_inv[9] * diff3;
		tmp += c_covar_inv[13] * diff4;
		tmp += c_covar_inv[18] * diff5;
		z_score += tmp * diff3;

		tmp = 0;
		tmp += c_covar_inv[10] * diff0;
		tmp += c_covar_inv[11] * diff1;
		tmp += c_covar_inv[12] * diff2;
		tmp += c_covar_inv[13] * diff3;
		tmp += c_covar_inv[14] * diff4;
		tmp += c_covar_inv[19] * diff5;
		z_score += tmp * diff4;

		tmp = 0;
		tmp += c_covar_inv[15] * diff0;
		tmp += c_covar_inv[16] * diff1;
		tmp += c_covar_inv[17] * diff2;
		tmp += c_covar_inv[18] * diff3;
		tmp += c_covar_inv[19] * diff4;
		tmp += c_covar_inv[20] * diff5;
		z_score += tmp * diff5;

		z_score = sqrtf(z_score);
		weight = z_score < trunc_sigma ? 1 : 0;

	#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
		d_o_weight[idx] = weight;
		const int covar_base = idx * 21;
		d_o_weighted_covar[covar_base + 0] = weight * diff0 * diff0;
		d_o_weighted_covar[covar_base + 1] = weight * diff1 * diff0;
		d_o_weighted_covar[covar_base + 2] = weight * diff1 * diff1;
		d_o_weighted_covar[covar_base + 3] = weight * diff2 * diff0;
		d_o_weighted_covar[covar_base + 4] = weight * diff2 * diff1;
		d_o_weighted_covar[covar_base + 5] = weight * diff2 * diff2;
		d_o_weighted_covar[covar_base + 6] = weight * diff3 * diff0;
		d_o_weighted_covar[covar_base + 7] = weight * diff3 * diff1;
		d_o_weighted_covar[covar_base + 8] = weight * diff3 * diff2;
		d_o_weighted_covar[covar_base + 9] = weight * diff3 * diff3;
		d_o_weighted_covar[covar_base + 10] = weight * diff4 * diff0;
		d_o_weighted_covar[covar_base + 11] = weight * diff4 * diff1;
		d_o_weighted_covar[covar_base + 12] = weight * diff4 * diff2;
		d_o_weighted_covar[covar_base + 13] = weight * diff4 * diff3;
		d_o_weighted_covar[covar_base + 14] = weight * diff4 * diff4;
		d_o_weighted_covar[covar_base + 15] = weight * diff5 * diff0;
		d_o_weighted_covar[covar_base + 16] = weight * diff5 * diff1;
		d_o_weighted_covar[covar_base + 17] = weight * diff5 * diff2;
		d_o_weighted_covar[covar_base + 18] = weight * diff5 * diff3;
		d_o_weighted_covar[covar_base + 19] = weight * diff5 * diff4;
		d_o_weighted_covar[covar_base + 20] = weight * diff5 * diff5;
	#endif
	}

	stats[0] = weight;
	stats[1] = weight * s0;
	stats[2] = weight * s1;
	stats[3] = weight * s2;
	stats[4] = weight * s3;
	stats[5] = weight * s4;
	stats[6] = weight * s5;
	stats[7] = weight * diff0 * diff0;
	stats[8] = weight * diff1 * diff0;
	stats[9] = weight * diff1 * diff1;
	stats[10] = weight * diff2 * diff0;
	stats[11] = weight * diff2 * diff1;
	stats[12] = weight * diff2 * diff2;
	stats[13] = weight * diff3 * diff0;
	stats[14] = weight * diff3 * diff1;
	stats[15] = weight * diff3 * diff2;
	stats[16] = weight * diff3 * diff3;
	stats[17] = weight * diff4 * diff0;
	stats[18] = weight * diff4 * diff1;
	stats[19] = weight * diff4 * diff2;
	stats[20] = weight * diff4 * diff3;
	stats[21] = weight * diff4 * diff4;
	stats[22] = weight * diff5 * diff0;
	stats[23] = weight * diff5 * diff1;
	stats[24] = weight * diff5 * diff2;
	stats[25] = weight * diff5 * diff3;
	stats[26] = weight * diff5 * diff4;
	stats[27] = weight * diff5 * diff5;

	for (int c = 0; c < 28; c++)
		stats[c] = rg_warp_reduce_sum(stats[c]);

	if (lane == 0) {
		for (int c = 0; c < 28; c++)
			s_warp_partials[warp_id * MAX_STATS_DIMS + c] = stats[c];
	}
	__syncthreads();

	if (warp_id == 0) {
		for (int c = 0; c < 28; c++) {
			float block_val = (lane < n_warps) ? s_warp_partials[lane * MAX_STATS_DIMS + c] : 0.0f;
			block_val = rg_warp_reduce_sum(block_val);
			if (lane == 0)
				d_o_partial_stats[blockIdx.x * MAX_STATS_DIMS + c] = block_val;
		}
	}
}

__global__ static void reduce_partial_stats(float* d_partial_stats, float* d_o_stats, int n_blocks, int stats_dims) {
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp_id = tid >> 5;
	const int n_warps = blockDim.x >> 5;
	__shared__ float s_warp_partials[8 * MAX_STATS_DIMS];

	for (int c = 0; c < stats_dims; c++) {
		float sum = 0.0f;
		for (int i = tid; i < n_blocks; i += blockDim.x)
			sum += d_partial_stats[i * MAX_STATS_DIMS + c];

		sum = rg_warp_reduce_sum(sum);
		if (lane == 0)
			s_warp_partials[warp_id * MAX_STATS_DIMS + c] = sum;
		__syncthreads();

		if (warp_id == 0) {
			float block_sum = (lane < n_warps) ? s_warp_partials[lane * MAX_STATS_DIMS + c] : 0.0f;
			block_sum = rg_warp_reduce_sum(block_sum);
			if (lane == 0)
				d_o_stats[c] = block_sum;
		}
		__syncthreads();
	}
}



int fit_robust_gaussian_device_scaled(
	float* d_space, float space_scale, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters) {
	if (dims > MAX_DIMS)
		throw;

	float* d_weight = nullptr;
	float* d_weighted_covar = nullptr;

	float* ht_weight; // host temp memory for weight. size=1
	float* ht_mean; // host temp memory for mean. size=dims
	float* ht_covar; // host temp memory for mean. size=dims*dims
	float* ht_covar_inv; // host temp memory for mean. size=dims*dims
	double* ht_covar_full; // host temp memory for mean. size=dims*dims
	double* ht_covar_inv_full; // host temp memory for mean. size=dims*dims
	float* ht_stats; // host temp memory for fused sums. size=1+dims+dims_covar

	const int dims_covar = (dims*dims + dims) / 2;
	const int stats_dims = 1 + dims + dims_covar;

	ensure_host_buffer_capacity(s_ht_weight, s_ht_weight_capacity, 1);
	ensure_host_buffer_capacity(s_ht_mean, s_ht_mean_capacity, dims);
	ensure_host_buffer_capacity(s_ht_covar, s_ht_covar_capacity, dims_covar);
	ensure_host_buffer_capacity(s_ht_covar_inv, s_ht_covar_inv_capacity, dims_covar);
	ensure_host_buffer_capacity(s_ht_covar_full, s_ht_covar_full_capacity, dims*dims);
	ensure_host_buffer_capacity(s_ht_covar_inv_full, s_ht_covar_inv_full_capacity, dims*dims);
	ensure_host_buffer_capacity(s_ht_stats, s_ht_stats_capacity, stats_dims);

	ht_weight = s_ht_weight;
	ht_mean = s_ht_mean;
	ht_covar = s_ht_covar;
	ht_covar_inv = s_ht_covar_inv;
	ht_covar_full = s_ht_covar_full;
	ht_covar_inv_full = s_ht_covar_inv_full;
	ht_stats = s_ht_stats;

	*ht_weight = 0;
	for (int d = 0; d < dims; d++)
		ht_mean[d] = h_io_mean[d];
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			ht_covar[(d1*d1 + d1) / 2 + d2] = h_io_covar[d1*dims + d2];


	// allocate device buffer
	int N_reduce_sum_ext = 0;
	for (int N_remain = N; N_remain > 1; N_remain = DIV_CEIL(N_remain, 2 * N_THREADS))
		N_reduce_sum_ext += DIV_CEIL(N_remain, 2 * N_THREADS);

	const int n_partial_blocks = DIV_CEIL(N, N_THREADS);
	const size_t required_partial_stats_bytes = (size_t)n_partial_blocks * (size_t)MAX_STATS_DIMS * sizeof(float);
	const size_t required_reduced_stats_bytes = (size_t)stats_dims * sizeof(float);
	ensure_device_buffer_capacity(s_d_partial_stats, s_partial_stats_capacity_bytes, required_partial_stats_bytes);
	ensure_device_buffer_capacity(s_d_reduced_stats, s_reduced_stats_capacity_bytes, required_reduced_stats_bytes);

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	const size_t required_weight_bytes = (size_t)N * sizeof(float);
	const size_t required_weighted_covar_bytes = (size_t)N * (size_t)dims_covar * sizeof(float);
	ensure_device_buffer_capacity(s_d_weight, s_weight_capacity_bytes, required_weight_bytes);
	ensure_device_buffer_capacity(s_d_weighted_covar, s_weighted_covar_capacity_bytes, required_weighted_covar_bytes);
	d_weight = s_d_weight;
	d_weighted_covar = s_d_weighted_covar;
#endif
	gpuErrchk;

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	float* d_weighted_LW_b2;
	float* ht_LW_b2; // host temp memory for LW_b2. size=1
	ensure_host_buffer_capacity(s_ht_LW_b2, s_ht_LW_b2_capacity, 1);
	ht_LW_b2 = s_ht_LW_b2;
	const size_t required_weighted_LW_b2_bytes = (size_t)(N + N_reduce_sum_ext) * sizeof(float);
	ensure_device_buffer_capacity(s_d_weighted_LW_b2, s_weighted_LW_b2_capacity_bytes, required_weighted_LW_b2_bytes);
	d_weighted_LW_b2 = s_d_weighted_LW_b2;
#endif

	if (used_iters != NULL)
		*used_iters = 0;

	// start iteration
	int iter;
	bool result_is_reliable = true;
	for (iter = 0; iter < max_iters; iter++) {

		cudaMemcpyToSymbol(c_mean, ht_mean, dims * sizeof(float));


		// process covar
		covar_half_to_full(ht_covar, ht_covar_full, dims);

		double covar_det; // important to have det double precision, since it is usually very small

		covar_det = determinant(ht_covar_full, dims);
		//printf("covar det = %.20lf\n", covar_det);

		//regularize covar
		if (iter > 0 && covar_reg_lambda > 0) {
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
			cudaMemcpyToSymbol(c_covar, ht_covar, dims_covar * sizeof(float));
			gpuErrchk;
			compute_LW_b2 << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_weight, d_weighted_covar, d_weighted_LW_b2, N, dims);
			gpuErrchk;
			reduce_vector_sum<N_THREADS>(d_weighted_LW_b2, ht_LW_b2, N, 1);
			gpuErrchk;
			(*ht_LW_b2) /= SQR(*ht_weight);

			regularize_covar_LW(ht_covar_full, ht_covar_full, (double)(*ht_LW_b2), dims);
#else
			regularize_covar_LW_given_lambda(ht_covar_full, ht_covar_full, covar_reg_lambda, dims);
#endif
		}
		//for (int d = 0; d < dims; d++)printf("%f ", ht_covar_full[d*dims + d]); printf("\n");
		covar_det = inverse(ht_covar_full, ht_covar_inv_full, dims);
		if (covar_det <= 0) {
			result_is_reliable = false;
			break;
		}

		covar_full_to_half(ht_covar_full, ht_covar, dims);
		covar_full_to_half(ht_covar_inv_full, ht_covar_inv, dims);


		cudaMemcpyToSymbol(c_covar_inv, ht_covar_inv, dims_covar * sizeof(float));
		gpuErrchk;

		// compute displacement and fused statistics
		float prev_density = *ht_weight / N;
		if (dims == 6)
			e_step_6_partial_reduce << <n_partial_blocks, N_THREADS >> > (d_space, d_weight, d_weighted_covar, s_d_partial_stats, trunc_sigma, N, space_scale);
		else
			e_step_partial_reduce << <n_partial_blocks, N_THREADS >> > (d_space, d_weight, d_weighted_covar, s_d_partial_stats, trunc_sigma, N, dims, space_scale);
		gpuErrchk;
		reduce_partial_stats << <1, N_THREADS >> > (s_d_partial_stats, s_d_reduced_stats, n_partial_blocks, stats_dims);
		gpuErrchk;
		cudaMemcpy(ht_stats, s_d_reduced_stats, stats_dims * sizeof(float), cudaMemcpyDeviceToHost);
		*ht_weight = ht_stats[0];
		gpuErrchk;

		if (!isfinite(*ht_weight)) {
			result_is_reliable = false;
			break;
		}

		float density_change = abs(*ht_weight / N - prev_density);

		if (density_change < epsilon) {
			result_is_reliable = true;
			break;
		}

		// update mean and covar only if no converge.
		// otherwise, mean and covar(reged) used in pervious iteration will be returned.

		for (int d = 0; d < dims; d++)
			ht_mean[d] = ht_stats[1 + d] / *ht_weight;

		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				ht_covar[(d1*d1 + d1) / 2 + d2] = ht_stats[1 + dims + (d1*d1 + d1) / 2 + d2] / (*ht_weight); // no -1 since it will be regularized later
					//ht_covar[(d1*d1 + d1) / 2 + d2] /= (*ht_weight - 1); // -1 for small sample number correction

		//printf("iter %d, density = %f, change = %f, covar_det = %e\n", iter, *ht_weight / N, density_change, covar_det);
		//for (int d = 0; d < dims; d++)printf("%f ", ht_covar[(d*d + d) / 2 + d]);printf("\n");
	}



	// update confidence & used_iter
	if (result_is_reliable) {
		if (h_o_density != NULL)
			*h_o_density = *ht_weight / N;
		if (used_iters != NULL)
			*used_iters = iter;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++) {
				h_io_covar[d1*dims + d2] = ht_covar[(d1*d1 + d1) / 2 + d2];
				h_io_covar[d2*dims + d1] = h_io_covar[d1*dims + d2];
			}
		for (int d = 0; d < dims; d++)
			h_io_mean[d] = ht_mean[d];
	}

	gpuErrchk;

	if (result_is_reliable)
		return cudaSuccess;
	else
		return !cudaSuccess;
}

int fit_robust_gaussian_device(
	float* d_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters) {
	return fit_robust_gaussian_device_scaled(
		d_space, 1.0f, h_io_mean, h_io_covar,
		trunc_sigma, covar_reg_lambda, h_o_density, used_iters,
		N, dims, epsilon, max_iters);
}

int fit_robust_gaussian(
	float* h_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters) {

	const size_t required_space_bytes = (size_t)N * (size_t)dims * sizeof(float);
	ensure_device_buffer_capacity(s_d_space, s_space_capacity_bytes, required_space_bytes);
	cudaMemcpy(s_d_space, h_space, required_space_bytes, cudaMemcpyHostToDevice);
	gpuErrchk;

	return fit_robust_gaussian_device(
		s_d_space, h_io_mean, h_io_covar,
		trunc_sigma, covar_reg_lambda, h_o_density, used_iters,
		N, dims, epsilon, max_iters);
}
