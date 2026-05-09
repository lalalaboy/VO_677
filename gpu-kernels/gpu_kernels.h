#pragma once

#if defined(WIN32) || defined(_WIN32)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

//TODO: add const...

struct DepthOptimizeTiming {
	float cache_upload_ms = 0.0f;
	float fb_smooth_ms = 0.0f;
	float init_cost_ms = 0.0f;
	float rand_prop_ms = 0.0f;
	float global_prop_ms = 0.0f;
	float local_prop_ms = 0.0f;
	float update_rigidness_ms = 0.0f;
	float copy_back_ms = 0.0f;

	float total_ms() const {
		return cache_upload_ms + fb_smooth_ms + init_cost_ms + rand_prop_ms +
			global_prop_ms + local_prop_ms + update_rigidness_ms + copy_back_ms;
	}
};

DLL_EXPORT int meanshift_gpu(float* h_space, float kernel_var,
	float* h_io_mean, float* h_o_confidence, int* used_iters,
	bool use_external_init_mean, int N, int dims,
	float epsilon = 1e-5f, int max_iters = 100,
	int max_init_trials = 20, float good_init_confidence = 0.5f);
DLL_EXPORT int meanshift_gpu_device(float* d_space, float kernel_var,
	float* h_io_mean, float* h_o_confidence, int* used_iters,
	bool use_external_init_mean, int N, int dims,
	float epsilon = 1e-5f, int max_iters = 100,
	int max_init_trials = 20, float good_init_confidence = 0.5f);
DLL_EXPORT int scale_pose_pool_rvecs_gpu(float* d_poses, float scale, int N);
DLL_EXPORT int scale_pose_pool_gpu(float* d_poses, float scale, int N);
DLL_EXPORT int copy_pose_pool_from_device(float* h_o_poses, float* d_poses, int N);

DLL_EXPORT int fit_robust_gaussian(
	float* h_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters);
DLL_EXPORT int fit_robust_gaussian_device_original_order(
	float* d_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters);
DLL_EXPORT int fit_robust_gaussian_device_original_order_scaled(
	float* d_space, float space_scale, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters);

DLL_EXPORT int collect_p3p_instances(
	float* h_flows[], float* h_rigidnesses[],
	float* h_depth,
	float* h_K, float* h_Rs[], float* h_ts[],
	float* h_o_p2_map, float* h_o_p3_map,
	int N, int w, int h,
	int active_idx,
	float rigidness_thresh,
	float rigidness_sum_thresh,
	float sample_min_depth,
	float sample_max_depth,
	int max_trace_on_flow);

DLL_EXPORT int compact_p3p_instances(
	float* h_o_pts2, float* h_o_pts3,
	int* h_o_n_points, int max_output_points,
	int w, int h);
DLL_EXPORT int compact_p3p_instances_device(
	float** d_o_pts2, float** d_o_pts3,
	int* h_o_n_points, int max_output_points,
	int w, int h);

DLL_EXPORT int solve_batch_p3p_ap3p_gpu(float* h_p3s, float* h_p2s,
	float* h_o_rvecs, float* h_o_tvecs,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_lambdatwist_gpu(float* h_p3s, float* h_p2s,
	float* h_o_rvecs, float* h_o_tvecs,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_ap3p_gpu_device(float* d_p3s, float* d_p2s,
	float* h_o_rvecs, float* h_o_tvecs,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_lambdatwist_gpu_device(float* d_p3s, float* d_p2s,
	float* h_o_rvecs, float* h_o_tvecs,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_ap3p_gpu_device_compact(float* d_p3s, float* d_p2s,
	float* h_o_poses, int* h_o_n_poses,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_lambdatwist_gpu_device_compact(float* d_p3s, float* d_p2s,
	float* h_o_poses, int* h_o_n_poses,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_ap3p_gpu_device_compact_device(float* d_p3s, float* d_p2s,
	float** d_o_poses, int* h_o_n_poses,
	float* h_K, int N_pts, int N_poses);
DLL_EXPORT int solve_batch_p3p_lambdatwist_gpu_device_compact_device(float* d_p3s, float* d_p2s,
	float** d_o_poses, int* h_o_n_poses,
	float* h_K, int N_pts, int N_poses);

DLL_EXPORT int optimize_depth_gpu(
	float* h_flows[],
	float* h_rigidnesses[], float* h_o_rigidnesses[],
	float* h_depth_priors[], float* h_depth_prior_pconfs[],
	float* h_depth_prior_confs[], float* h_o_depth_prior_confs[],
	float* h_depth, float* h_o_depth,
	float* h_K, float* h_Rs[], float* h_ts[],
	float* h_dp_Rs[], float* h_dp_ts[],
	float abs_resize_factor,
	int N, int N_dp, int w, int h, float basefocal,
	int n_rand_samples, int global_prop_step, int local_prop_width,
	float lambda, float omega, float disp_delta, float delta,
	bool fb_smooth, float s0_ems_prob, float no_change_prob,
	float range_factor,
	bool update_rigidness_only,
	DepthOptimizeTiming* timing = nullptr);

DLL_EXPORT int align_frame_init_gpu(
	float* h_images[],
	float* h_depths[],
	float* h_weights[],
	float* h_K,
	float vbf, float crw,
	int N, int w, int h);

DLL_EXPORT int align_frame_eval_gpu(
	int ref_fid,
	int tar_fid,
	const float* h_params_ref,
	const float* h_params_tar,
	float* h_o_residual, float* h_o_jacobian,
	const bool apply_weights = true);
