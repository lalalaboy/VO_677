#include "utils.h"
#include "gpu_kernels.h"
#include "reduce_vector_sum.h"

#define MAX_DIMS 16

#define N_THREADS 256

__constant__ static float c_mean[MAX_DIMS];


template <bool compute_weight_only = false>
__global__ static void compute_weighted_space(
	float* d_space, float kernel_var, float* d_o_weight, float* d_o_weighted_space, int N, int dims) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float s_mean[MAX_DIMS];
	if (threadIdx.x < dims)
		s_mean[threadIdx.x] = c_mean[threadIdx.x];
	__syncthreads();

	if (idx < N) {
		float l2_distance_sqr = 0;
		float weight = 0;

		for (int d = 0; d < dims; d++)
			l2_distance_sqr += SQR(d_space[idx*dims + d] - s_mean[d]);

		weight = expf(-l2_distance_sqr / (2 * kernel_var));
		d_o_weight[idx] = weight;
		if (!compute_weight_only) {
			for (int d = 0; d < dims; d++)
				d_o_weighted_space[idx*dims + d] = d_space[idx*dims + d] * weight;
		}
	}
}


int meanshift_gpu(float* h_space, float kernel_var,
	float* h_io_mean, float* h_o_confidence, int* used_iters,
	bool use_external_init_mean, int N, int dims,
	float epsilon, int max_iters,
	int max_init_trials, float good_init_confidence) {

	static float* d_space = NULL;
	static float* d_weight = NULL;
	static float* d_weighted_space = NULL;
	static int d_space_cap = 0; // in number of float entries
	static int d_weight_cap = 0; // in number of float entries
	static int d_weighted_space_cap = 0; // in number of float entries
	static float* ht_mean = NULL; // host temp memory for mean. size=dims
	static int ht_mean_cap = 0;
	float ht_weight[1];

	const int space_need = N * dims;
	if (space_need > d_space_cap) {
		if (d_space) cudaFree(d_space);
		cudaMalloc((void**)&d_space, space_need * sizeof(float));
		d_space_cap = space_need;
	}

	// copy space data
	cudaMemcpy(d_space, h_space, N * dims * sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk;

	// allocate device buffer (weight map & weighted space map)
	int N_reduce_sum_ext = 0;
	for (int N_remain = N; N_remain > 1; N_remain = DIV_CEIL(N_remain, 2 * N_THREADS))
		N_reduce_sum_ext += DIV_CEIL(N_remain, 2 * N_THREADS);
	const int weight_need = N + N_reduce_sum_ext;
	const int weighted_space_need = (N + N_reduce_sum_ext) * dims;
	if (weight_need > d_weight_cap) {
		if (d_weight) cudaFree(d_weight);
		cudaMalloc((void**)&d_weight, weight_need * sizeof(float));
		d_weight_cap = weight_need;
	}
	if (weighted_space_need > d_weighted_space_cap) {
		if (d_weighted_space) cudaFree(d_weighted_space);
		cudaMalloc((void**)&d_weighted_space, weighted_space_need * sizeof(float));
		d_weighted_space_cap = weighted_space_need;
	}
	if (dims > ht_mean_cap) {
		if (ht_mean) delete[] ht_mean;
		ht_mean = new float[dims];
		ht_mean_cap = dims;
	}
	gpuErrchk;

	// init mean with external given
	if (use_external_init_mean)
		cudaMemcpyToSymbol(c_mean, h_io_mean, dims * sizeof(float));
	else { // init mean with trial of maximum confidence

		float best_init_confidence = 0;
		int best_init_sample_idx = -1;
		for (int trial = 0; trial < max_init_trials; trial++) {
			int idx_rand = (rand() % N);

			cudaMemcpyToSymbol(c_mean, &h_space[idx_rand*dims], dims * sizeof(float));
			gpuErrchk;

			// weight each sample
			compute_weighted_space<true> << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_space, kernel_var, d_weight, NULL, N, dims);
			// sum up weight
			reduce_vector_sum<N_THREADS>(d_weight, ht_weight, N, 1);
			gpuErrchk;

			if (*ht_weight > best_init_confidence) {
				best_init_confidence = *ht_weight;
				best_init_sample_idx = idx_rand;
			}

			if (best_init_confidence > good_init_confidence*N)
				break;
		}
		cudaMemcpyToSymbol(c_mean, &h_space[best_init_sample_idx*dims], dims * sizeof(float));
		gpuErrchk;
	}

	if (used_iters != NULL)
		*used_iters = 0;

	// start iteration
	for (int iter = 0; iter < max_iters; iter++) {

		compute_weighted_space << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_space, kernel_var, d_weight, d_weighted_space, N, dims);
		reduce_vector_sum<N_THREADS>(d_weight, ht_weight, N, 1);
		reduce_vector_sum<N_THREADS>(d_weighted_space, ht_mean, N, dims);
		gpuErrchk;

		for (int d = 0; d < dims; d++)
			ht_mean[d] = ht_mean[d] / *ht_weight;

		// update confidence & used_iter
		if (h_o_confidence != NULL)
			*h_o_confidence = *ht_weight / N;
		if (used_iters != NULL)
			*used_iters = iter + 1;

		// compute displacement
		float displacement = 0;
		for (int d = 0; d < dims; d++)
			displacement += SQR(h_io_mean[d] - ht_mean[d]);
		displacement = sqrt(displacement);

		// after compute displacement, assign back mean
		for (int d = 0; d < dims; d++)
			h_io_mean[d] = ht_mean[d];

		if (displacement < epsilon)
			break;

		cudaMemcpyToSymbol(c_mean, h_io_mean, dims * sizeof(float));
		gpuErrchk;
	}

	return cudaSuccess;
}
