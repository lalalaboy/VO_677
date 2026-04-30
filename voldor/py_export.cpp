#include "py_export.h"
#include "voldor.h"
#include <iterator>

int py_voldor_wrapper(
	// inputs
	const float* flows_pt, const float* disparity_pt, const float* disparity_pconf_pt,
	const float* depth_priors_pt, const float* depth_prior_poses_pt, const float* depth_prior_pconfs_pt,
	const float fx, const float fy, const float cx, const float cy, const float basefocal,
	const int N, const int N_dp, const int w, const int h,
	const char* config_pt,
	// outputs
	int& n_registered, float* poses_pt, float* poses_covar_pt, float* depth_pt, float* depth_conf_pt,
	float& sampling_collection_ms_total, float& p3p_computing_ms_total,
	float& meanshift_ms_total, float& gu_fit_ms_total,
	float& depth_cache_upload_ms_total, float& depth_fb_smooth_ms_total,
	float& depth_init_cost_ms_total, float& depth_rand_prop_ms_total,
	float& depth_global_prop_ms_total, float& depth_local_prop_ms_total,
	float& depth_update_rigidness_ms_total, float& depth_copy_back_ms_total,
	float& total_runtime_ms_per_frame,
	int& pose_opt_timed_calls, int& pose_opt_gu_fit_calls,
	int& depth_opt_timed_calls) {

	Config cfg;
	std::istringstream iss(config_pt);
	std::vector<std::string> cfg_strs(
		std::istream_iterator<std::string>{iss},
		std::istream_iterator<std::string>());
	cfg.fx = fx;
	cfg.cx = cx;
	cfg.fy = fy;
	cfg.cy = cy;
	cfg.basefocal = basefocal;
	cfg.read_config(cfg_strs);


	vector<Mat> flows;
	Mat disparity;
	Mat disparity_pconf;
	vector<Mat> depth_priors;
	vector<Vec6f> depth_prior_poses;
	vector<Mat> depth_priors_pconfs;

	for (int i = 0; i < N; i++) {
		flows.push_back(Mat(Size(w, h), CV_32FC2, (void*)(flows_pt + i * w*h * 2)));
	}

	if (disparity_pt)
		disparity = Mat(Size(w, h), CV_32F, (void*)disparity_pt);
	if (disparity_pconf_pt)
		disparity_pconf = Mat(Size(w, h), CV_32F, (void*)disparity_pconf_pt);

	for (int i = 0; i < N_dp; i++) {
		depth_priors.push_back(Mat(Size(w, h), CV_32F, (void*)(depth_priors_pt + i * w*h)));
		depth_prior_poses.push_back(Vec6f(depth_prior_poses_pt + i * 6));
		if (depth_prior_pconfs_pt)
			depth_priors_pconfs.push_back(Mat(Size(w, h), CV_32F, (void*)(depth_prior_pconfs_pt + i * w*h)));
	}


	VOLDOR voldor(cfg);
	voldor.init(flows, disparity, disparity_pconf, depth_priors, depth_prior_poses, depth_priors_pconfs);
	voldor.solve();
	sampling_collection_ms_total = voldor.pose_opt_timing_total.sampling_collection_ms;
	p3p_computing_ms_total = voldor.pose_opt_timing_total.p3p_computing_ms;
	meanshift_ms_total = voldor.pose_opt_timing_total.meanshift_ms;
	gu_fit_ms_total = voldor.pose_opt_timing_total.gu_fit_ms;
	depth_cache_upload_ms_total = voldor.depth_opt_timing_total.cache_upload_ms;
	depth_fb_smooth_ms_total = voldor.depth_opt_timing_total.fb_smooth_ms;
	depth_init_cost_ms_total = voldor.depth_opt_timing_total.init_cost_ms;
	depth_rand_prop_ms_total = voldor.depth_opt_timing_total.rand_prop_ms;
	depth_global_prop_ms_total = voldor.depth_opt_timing_total.global_prop_ms;
	depth_local_prop_ms_total = voldor.depth_opt_timing_total.local_prop_ms;
	depth_update_rigidness_ms_total = voldor.depth_opt_timing_total.update_rigidness_ms;
	depth_copy_back_ms_total = voldor.depth_opt_timing_total.copy_back_ms;
	pose_opt_timed_calls = voldor.pose_opt_timed_calls;
	pose_opt_gu_fit_calls = voldor.pose_opt_gu_fit_calls;
	depth_opt_timed_calls = voldor.depth_opt_timed_calls;
	total_runtime_ms_per_frame = pose_opt_timed_calls > 0 ?
		voldor.pose_opt_timing_total.total_ms() / pose_opt_timed_calls : 0.0f;

	n_registered = voldor.n_flows;

	for (int i = 0; i < voldor.n_flows; i++) {
		if (poses_pt)
			memcpy(poses_pt + i * 6, voldor.cams[i].pose6().val, 6 * sizeof(float));
		if (poses_covar_pt)
			memcpy(poses_covar_pt + i * 6 * 6, voldor.cams[i].pose_covar.data, 6 * 6 * sizeof(float));
	}

	if (depth_pt)
		memcpy(depth_pt, voldor.depth.data, w*h * sizeof(float));

	if (depth_conf_pt) {
		Mat depth_conf = Mat::zeros(Size(w, h), CV_32F);
		for (int i = 0; i < voldor.n_flows; i++)
			depth_conf += voldor.rigidnesses[i];
		for (int i = 0; i < voldor.n_depth_priors; i++)
			depth_conf += voldor.depth_prior_confs[i];
		depth_conf /= (float)(voldor.n_flows + voldor.n_depth_priors);
		memcpy(depth_conf_pt, depth_conf.data, w*h * sizeof(float));
	}

	return 0;
}
