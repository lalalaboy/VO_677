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
static float* s_d_weight = nullptr;
static float* s_d_weighted_covar = nullptr;
static float* s_d_partial_stats = nullptr;
static float* s_d_reduced_stats = nullptr;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static float* s_d_weighted_LW_b2 = nullptr;
#endif

static size_t s_space_capacity_bytes = 0;
static size_t s_weight_capacity_bytes = 0;
static size_t s_weighted_covar_capacity_bytes = 0;
static size_t s_partial_stats_capacity_bytes = 0;
static size_t s_reduced_stats_capacity_bytes = 0;
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
static size_t s_weighted_LW_b2_capacity_bytes = 0;
#endif

constexpr int MAX_COVAR_DIMS = (MAX_DIMS * MAX_DIMS + MAX_DIMS) / 2;
constexpr int MAX_STATS_DIMS = 1 + MAX_DIMS + MAX_COVAR_DIMS; // weight + mean + covar

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

__device__ __forceinline__ static float warp_reduce_sum(float val) {
	for (int offset = 16; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xffffffff, val, offset);
	return val;
}

// computes per-sample values and partially reduces weight/mean/covar per block
__global__ static void e_step_partial_reduce( // c_mean, c_covar_inv are needed
	float* d_space,
	float* d_o_weight, float* d_o_weighted_covar, float* d_o_partial_stats,
	float trunc_sigma,
	int N, int dims) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp_id = tid >> 5;
	const int n_warps = blockDim.x >> 5;
	const int covar_dims = (dims*dims + dims) / 2;
	const int stats_dims = 1 + dims + covar_dims;
	__shared__ float s_warp_partials[8 * MAX_STATS_DIMS]; // N_THREADS=256 => max 8 warps

	float diff[MAX_DIMS];
	float weight = 0.0f;
	float stats[MAX_STATS_DIMS]{ 0 };

	if (idx < N) {
		for (int d = 0; d < dims; d++)
			diff[d] = d_space[idx*dims + d] - c_mean[d];


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

		d_o_weight[idx] = weight;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				d_o_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] = weight * diff[d1] * diff[d2];
	}

	stats[0] = weight;
	for (int d = 0; d < dims; d++)
		stats[1 + d] = weight * (idx < N ? d_space[idx*dims + d] : 0.0f);
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			stats[1 + dims + (d1*d1 + d1) / 2 + d2] = weight * (idx < N ? diff[d1] * diff[d2] : 0.0f);

	for (int c = 0; c < stats_dims; c++)
		stats[c] = warp_reduce_sum(stats[c]);

	if (lane == 0) {
		for (int c = 0; c < stats_dims; c++)
			s_warp_partials[warp_id * MAX_STATS_DIMS + c] = stats[c];
	}
	__syncthreads();

	float block_val = 0.0f;
	if (warp_id == 0) {
		for (int c = 0; c < stats_dims; c++) {
			block_val = (lane < n_warps) ? s_warp_partials[lane * MAX_STATS_DIMS + c] : 0.0f;
			block_val = warp_reduce_sum(block_val);
			if (lane == 0)
				d_o_partial_stats[blockIdx.x * MAX_STATS_DIMS + c] = block_val;
		}
	}
}

// second-stage reduction across blocks, using warp + block reduction
__global__ static void reduce_partial_stats(float* d_partial_stats, float* d_o_stats, int n_blocks, int stats_dims) {
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warp_id = tid >> 5;
	const int n_warps = blockDim.x >> 5;
	__shared__ float s_warp_partials[8 * MAX_STATS_DIMS]; // N_THREADS=256 => max 8 warps

	for (int c = 0; c < stats_dims; c++) {
		float sum = 0.0f;
		for (int i = tid; i < n_blocks; i += blockDim.x)
			sum += d_partial_stats[i * MAX_STATS_DIMS + c];

		sum = warp_reduce_sum(sum);
		if (lane == 0)
			s_warp_partials[warp_id * MAX_STATS_DIMS + c] = sum;
		__syncthreads();

		if (warp_id == 0) {
			float block_sum = (lane < n_warps) ? s_warp_partials[lane * MAX_STATS_DIMS + c] : 0.0f;
			block_sum = warp_reduce_sum(block_sum);
			if (lane == 0)
				d_o_stats[c] = block_sum;
		}
		__syncthreads();
	}
}



int fit_robust_gaussian(
	float* h_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters) {
	if (dims > MAX_DIMS)
		throw;

	float* d_space = nullptr;
	float* d_weight = nullptr;
	float* d_weighted_covar = nullptr;
	float* d_partial_stats = nullptr;
	float* d_reduced_stats = nullptr;

	float* ht_weight; // host temp memory for weight. size=1
	float* ht_mean; // host temp memory for mean. size=dims
	float* ht_covar; // host temp memory for mean. size=dims*dims
	float* ht_covar_inv; // host temp memory for mean. size=dims*dims
	double* ht_covar_full; // host temp memory for mean. size=dims*dims
	double* ht_covar_inv_full; // host temp memory for mean. size=dims*dims
	float* ht_stats; // host temp memory for reduced stats. size=MAX_STATS_DIMS

	const int dims_covar = (dims*dims + dims) / 2;

	ht_weight = new float[1];
	ht_mean = new float[dims];
	ht_covar = new float[dims_covar];
	ht_covar_inv = new float[dims_covar];
	ht_covar_full = new double[dims*dims];
	ht_covar_inv_full = new double[dims*dims];
	ht_stats = new float[MAX_STATS_DIMS];

	*ht_weight = 0;
	for (int d = 0; d < dims; d++)
		ht_mean[d] = h_io_mean[d];
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			ht_covar[(d1*d1 + d1) / 2 + d2] = h_io_covar[d1*dims + d2];


	// reuse cached device buffers and only grow when needed
	const size_t required_space_bytes = (size_t)N * (size_t)dims * sizeof(float);
	ensure_device_buffer_capacity(s_d_space, s_space_capacity_bytes, required_space_bytes);
	d_space = s_d_space;

	// copy space data
	cudaMemcpy(d_space, h_space, required_space_bytes, cudaMemcpyHostToDevice);
	gpuErrchk;

	// allocate device buffer (weight map & weighted space map)
	int N_reduce_sum_ext = 0;
	for (int N_remain = N; N_remain > 1; N_remain = DIV_CEIL(N_remain, 2 * N_THREADS))
		N_reduce_sum_ext += DIV_CEIL(N_remain, 2 * N_THREADS);

	const size_t required_weight_bytes = (size_t)(N + N_reduce_sum_ext) * sizeof(float);
	const size_t required_weighted_covar_bytes = (size_t)(N + N_reduce_sum_ext) * (size_t)dims_covar * sizeof(float);
	const int n_blocks = DIV_CEIL(N, N_THREADS);
	const int stats_dims = 1 + dims + dims_covar;
	const size_t required_partial_stats_bytes = (size_t)n_blocks * MAX_STATS_DIMS * sizeof(float);
	const size_t required_reduced_stats_bytes = (size_t)MAX_STATS_DIMS * sizeof(float);
	ensure_device_buffer_capacity(s_d_weight, s_weight_capacity_bytes, required_weight_bytes);
	ensure_device_buffer_capacity(s_d_weighted_covar, s_weighted_covar_capacity_bytes, required_weighted_covar_bytes);
	ensure_device_buffer_capacity(s_d_partial_stats, s_partial_stats_capacity_bytes, required_partial_stats_bytes);
	ensure_device_buffer_capacity(s_d_reduced_stats, s_reduced_stats_capacity_bytes, required_reduced_stats_bytes);
	d_weight = s_d_weight;
	d_weighted_covar = s_d_weighted_covar;
	d_partial_stats = s_d_partial_stats;
	d_reduced_stats = s_d_reduced_stats;
	gpuErrchk;

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	float* d_weighted_LW_b2;
	float* ht_LW_b2; // host temp memory for LW_b2. size=1
	ht_LW_b2 = new float[1];
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

		// compute displacement
		float prev_density = *ht_weight / N;
		e_step_partial_reduce << <n_blocks, N_THREADS >> > (d_space, d_weight, d_weighted_covar, d_partial_stats, trunc_sigma, N, dims);
		reduce_partial_stats << <1, N_THREADS >> > (d_partial_stats, d_reduced_stats, n_blocks, stats_dims);
		cudaMemcpy(ht_stats, d_reduced_stats, stats_dims * sizeof(float), cudaMemcpyDeviceToHost);
		gpuErrchk;
		*ht_weight = ht_stats[0];

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
			ht_mean[d] = ht_stats[1 + d];
		for (int d = 0; d < dims_covar; d++)
			ht_covar[d] = ht_stats[1 + dims + d];

		for (int d = 0; d < dims; d++)
			ht_mean[d] /= *ht_weight;

		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				ht_covar[(d1*d1 + d1) / 2 + d2] /= (*ht_weight); // no -1 since it will be regularized later
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

	delete[] ht_weight;
	delete[] ht_mean;
	delete[] ht_covar;
	delete[] ht_covar_inv;
	delete[] ht_covar_full;
	delete[] ht_covar_inv_full;
	delete[] ht_stats;

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	delete[] ht_LW_b2;
#endif
	gpuErrchk;

	if (result_is_reliable)
		return cudaSuccess;
	else
		return !cudaSuccess;
}
