#include "utils.h"
#include "gpu_kernels.h"
#include "rodrigues.h"
#include "../lambdatwist/lambdatwist_p4p.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <limits.h>

#define N_THREADS 32

__constant__ static float _fx, _fy, _cx, _cy;
static float* d_p2s_cache = NULL;
static float* d_p3s_cache = NULL;
static float* d_rvecs_cache = NULL;
static float* d_tvecs_cache = NULL;
static curandState* d_rand_states_cache = NULL;
static int* d_pose_flags_cache = NULL;
static int* d_pose_prefix_cache = NULL;
static float* d_pose_compact_cache = NULL;
static int pts_cap = 0;
static int poses_cap = 0;

static int grow_cache_capacity(int current_cap, int required) {
	int next_cap = current_cap > 0 ? current_cap : 1;
	while (next_cap < required) {
		int increment = next_cap / 2;
		if (increment < 1)
			increment = 1;
		if (next_cap > INT_MAX - increment)
			return required;
		next_cap += increment;
	}
	return next_cap;
}


__global__ static void solve(float* d_p2s, float* d_p3s, float* d_rvecs, float* d_tvecs, curandState* d_rand_states, int N_pts, int N_poses) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N_poses) {
		int i1 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i2 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i3 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i4 = curand_uniform(&d_rand_states[idx])*N_pts;
		float R[3][3], t[3];

		bool success = lambdatwist_p4p<float, float, 5>(
			&d_p2s[i1 * 2], &d_p2s[i2 * 2], &d_p2s[i3 * 2], &d_p2s[i4 * 2],
			&d_p3s[i1 * 3], &d_p3s[i2 * 3], &d_p3s[i3 * 3], &d_p3s[i4 * 3],
			_fx, _fy, _cx, _cy,
			R, t);


		if (!success) {
			d_rvecs[idx * 3 + 0] = CUDART_NAN_F; d_rvecs[idx * 3 + 1] = CUDART_NAN_F; d_rvecs[idx * 3 + 2] = CUDART_NAN_F;
			d_tvecs[idx * 3 + 0] = CUDART_NAN_F; d_tvecs[idx * 3 + 1] = CUDART_NAN_F; d_tvecs[idx * 3 + 2] = CUDART_NAN_F;
			return; // false
		}
		
		d_tvecs[idx * 3 + 0] = t[0];
		d_tvecs[idx * 3 + 1] = t[1];
		d_tvecs[idx * 3 + 2] = t[2];
		rodrigues(R, &d_rvecs[idx * 3]);
		
		
	}
}

__global__ static void init_rand_states(curandState* d_rand_states, int N) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < N)
		curand_init(RAND_SEED, idx, 0, &d_rand_states[idx]);
}

__global__ static void build_pose_valid_flags(float* d_rvecs, float* d_tvecs, int* d_o_flags, int N) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < N) {
		const float sum =
			d_rvecs[idx * 3 + 0] + d_rvecs[idx * 3 + 1] + d_rvecs[idx * 3 + 2] +
			d_tvecs[idx * 3 + 0] + d_tvecs[idx * 3 + 1] + d_tvecs[idx * 3 + 2];
		d_o_flags[idx] = isfinite(sum) ? 1 : 0;
	}
}

__global__ static void scatter_valid_poses(
	float* d_rvecs, float* d_tvecs,
	const int* d_flags, const int* d_prefix,
	float* d_o_poses, int N) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < N && d_flags[idx]) {
		const int out_idx = d_prefix[idx];
		d_o_poses[out_idx * 6 + 0] = d_rvecs[idx * 3 + 0];
		d_o_poses[out_idx * 6 + 1] = d_rvecs[idx * 3 + 1];
		d_o_poses[out_idx * 6 + 2] = d_rvecs[idx * 3 + 2];
		d_o_poses[out_idx * 6 + 3] = d_tvecs[idx * 3 + 0];
		d_o_poses[out_idx * 6 + 4] = d_tvecs[idx * 3 + 1];
		d_o_poses[out_idx * 6 + 5] = d_tvecs[idx * 3 + 2];
	}
}

static int compact_finite_poses(float* h_o_poses, int* h_o_n_poses, int N_poses) {
	if (N_poses <= 0) {
		if (h_o_n_poses)
			*h_o_n_poses = 0;
		return cudaSuccess;
	}

	const int grid_size = DIV_CEIL(N_poses, N_THREADS);
	build_pose_valid_flags << <grid_size, N_THREADS >> > (d_rvecs_cache, d_tvecs_cache, d_pose_flags_cache, N_poses);
	gpuErrchk;

	thrust::exclusive_scan(
		thrust::device_pointer_cast(d_pose_flags_cache),
		thrust::device_pointer_cast(d_pose_flags_cache + N_poses),
		thrust::device_pointer_cast(d_pose_prefix_cache));

	scatter_valid_poses << <grid_size, N_THREADS >> > (
		d_rvecs_cache, d_tvecs_cache, d_pose_flags_cache, d_pose_prefix_cache, d_pose_compact_cache, N_poses);
	gpuErrchk;

	int last_prefix = 0;
	int last_flag = 0;
	cudaMemcpy(&last_prefix, d_pose_prefix_cache + N_poses - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&last_flag, d_pose_flags_cache + N_poses - 1, sizeof(int), cudaMemcpyDeviceToHost);
	gpuErrchk;

	const int n_poses = last_prefix + last_flag;
	if (h_o_poses && n_poses > 0)
		cudaMemcpy(h_o_poses, d_pose_compact_cache, n_poses * 6 * sizeof(float), cudaMemcpyDeviceToHost);
	gpuErrchk;

	if (h_o_n_poses)
		*h_o_n_poses = n_poses;

	return cudaSuccess;
}


static int solve_batch_p3p_lambdatwist_gpu_impl(
	float* d_p3s, float* d_p2s,
	float* h_o_rvecs, float* h_o_tvecs,
	float* h_K, int N_pts, int N_poses) {

	// copy K to gpu constant memory
	static float cache_symbols[4] = { 0 };
	if (h_K) {
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[0], cache_symbols[0], _fx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[2], cache_symbols[2], _cx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[4], cache_symbols[1], _fy);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[5], cache_symbols[3], _cy);
	}
	if (N_poses > poses_cap) {
		const int new_poses_cap = grow_cache_capacity(poses_cap, N_poses);
		if (d_rvecs_cache) cudaFree(d_rvecs_cache);
		if (d_tvecs_cache) cudaFree(d_tvecs_cache);
		if (d_rand_states_cache) cudaFree(d_rand_states_cache);
		if (d_pose_flags_cache) cudaFree(d_pose_flags_cache);
		if (d_pose_prefix_cache) cudaFree(d_pose_prefix_cache);
		if (d_pose_compact_cache) cudaFree(d_pose_compact_cache);
		cudaMalloc((void**)&d_rvecs_cache, new_poses_cap * 3 * sizeof(float));
		cudaMalloc((void**)&d_tvecs_cache, new_poses_cap * 3 * sizeof(float));
		cudaMalloc((void**)&d_rand_states_cache, new_poses_cap * sizeof(curandState));
		cudaMalloc((void**)&d_pose_flags_cache, new_poses_cap * sizeof(int));
		cudaMalloc((void**)&d_pose_prefix_cache, new_poses_cap * sizeof(int));
		cudaMalloc((void**)&d_pose_compact_cache, new_poses_cap * 6 * sizeof(float));
		poses_cap = new_poses_cap;
	}
	gpuErrchk;

	init_rand_states << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_rand_states_cache, N_poses);
	gpuErrchk;

	// solve batch ap3p
	solve << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_p2s, d_p3s, d_rvecs_cache, d_tvecs_cache, d_rand_states_cache, N_pts, N_poses);
	gpuErrchk;

	// copy back R,t
	if (h_o_rvecs)
		cudaMemcpy(h_o_rvecs, d_rvecs_cache, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (h_o_tvecs)
		cudaMemcpy(h_o_tvecs, d_tvecs_cache, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	gpuErrchk;

	return cudaSuccess;
}

int solve_batch_p3p_lambdatwist_gpu(float* h_p3s, float* h_p2s, float* h_o_rvecs, float* h_o_tvecs, float* h_K, int N_pts, int N_poses) {
	if (N_pts > pts_cap) {
		const int new_pts_cap = grow_cache_capacity(pts_cap, N_pts);
		if (d_p2s_cache) cudaFree(d_p2s_cache);
		if (d_p3s_cache) cudaFree(d_p3s_cache);
		cudaMalloc((void**)&d_p2s_cache, new_pts_cap * 2 * sizeof(float));
		cudaMalloc((void**)&d_p3s_cache, new_pts_cap * 3 * sizeof(float));
		pts_cap = new_pts_cap;
	}
	cudaMemcpy(d_p2s_cache, h_p2s, N_pts * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p3s_cache, h_p3s, N_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk;
	return solve_batch_p3p_lambdatwist_gpu_impl(d_p3s_cache, d_p2s_cache, h_o_rvecs, h_o_tvecs, h_K, N_pts, N_poses);
}

int solve_batch_p3p_lambdatwist_gpu_device(float* d_p3s, float* d_p2s, float* h_o_rvecs, float* h_o_tvecs, float* h_K, int N_pts, int N_poses) {
	return solve_batch_p3p_lambdatwist_gpu_impl(d_p3s, d_p2s, h_o_rvecs, h_o_tvecs, h_K, N_pts, N_poses);
}

int solve_batch_p3p_lambdatwist_gpu_device_compact(float* d_p3s, float* d_p2s, float* h_o_poses, int* h_o_n_poses, float* h_K, int N_pts, int N_poses) {
	const int status = solve_batch_p3p_lambdatwist_gpu_impl(d_p3s, d_p2s, NULL, NULL, h_K, N_pts, N_poses);
	if (status != cudaSuccess)
		return status;
	return compact_finite_poses(h_o_poses, h_o_n_poses, N_poses);
}

int solve_batch_p3p_lambdatwist_gpu_device_compact_device(float* d_p3s, float* d_p2s, float** d_o_poses, int* h_o_n_poses, float* h_K, int N_pts, int N_poses) {
	const int status = solve_batch_p3p_lambdatwist_gpu_impl(d_p3s, d_p2s, NULL, NULL, h_K, N_pts, N_poses);
	if (status != cudaSuccess)
		return status;
	const int compact_status = compact_finite_poses(NULL, h_o_n_poses, N_poses);
	if (compact_status != cudaSuccess)
		return compact_status;
	if (d_o_poses)
		*d_o_poses = h_o_n_poses && *h_o_n_poses > 0 ? d_pose_compact_cache : NULL;
	return cudaSuccess;
}
