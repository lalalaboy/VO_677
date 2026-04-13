#pragma once

extern int py_voldor_wrapper(
	// inputs
	const float* flows, const float* disparity, const float* disparity_pconf,
	const float* depth_priors, const float* depth_prior_poses, const float* depth_prior_pconfs,
	const float fx, const float fy, const float cx, const float cy, const float basefocal,
	const int N, const int N_dp, const int w, const int h,
	const char* config,
	// outputs
	int& n_registered, float* poses, float* poses_covar, float* depth, float* depth_conf,
	float& sampling_collection_ms_total, float& p3p_computing_ms_total,
	float& meanshift_ms_total, float& gu_fit_ms_total,
	int& pose_opt_timed_calls, int& pose_opt_gu_fit_calls);
