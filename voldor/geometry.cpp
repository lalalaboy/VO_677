#include "geometry.h"
#include "../gpu-kernels/gpu_kernels.h"
#include "../lambdatwist/lambdatwist_p4p.h"

int optimize_camera_pose(vector<Mat> flows, vector<Mat> rigidnesses,
	Mat depth, vector<Camera>& cams,
	int n_flows, int active_idx, bool successive_pose,
	bool rg_refine,
	bool update_batch_instance, bool update_iter_instance,
	Config cfg) {

	const int w = flows[0].cols, h = flows[0].rows;
	const int pixel_count = w * h;
	auto pose_total_t0 = chrono::high_resolution_clock::now();
	auto time_stamp = pose_total_t0;

	vector<Point2f> pts2_local;
	vector<Point3f> pts3_local;
	thread_local vector<Point2f> pts2_reuse;
	thread_local vector<Point3f> pts3_reuse;
	vector<Point2f>& pts2 = cfg.pose_fast_mode > 0 ? pts2_reuse : pts2_local;
	vector<Point3f>& pts3 = cfg.pose_fast_mode > 0 ? pts3_reuse : pts3_local;
	if ((int)pts2.size() < pixel_count)
		pts2.resize(pixel_count);
	if ((int)pts3.size() < pixel_count)
		pts3.resize(pixel_count);
	// pts2 is related to frame(active_idx), pts3 is related to frame(active_idx-1).
	// Thus, the relative pose describe frame(active_idx-1)--[R|Rt]-->frame(active_idx).

	vector<float*> h_flows(n_flows);
	vector<float*> h_rigidnesses(n_flows);
	vector<float*> h_Rs(n_flows);
	vector<float*> h_ts(n_flows);

	for (int i = 0; i < n_flows; i++) {
		h_flows[i] = (float*) flows[i].data;
		h_rigidnesses[i] = (float*) rigidnesses[i].data;
		h_Rs[i] = (float*) cams[i].R.data;
		h_ts[i] = (float*) cams[i].t.data;
	}

	Mat pts2_map_local;
	Mat pts3_map_local;
	thread_local Mat pts2_map_reuse;
	thread_local Mat pts3_map_reuse;
	Mat& pts2_map = cfg.pose_fast_mode > 0 ? pts2_map_reuse : pts2_map_local;
	Mat& pts3_map = cfg.pose_fast_mode > 0 ? pts3_map_reuse : pts3_map_local;
	pts2_map.create(Size(w, h), CV_32FC2);
	pts3_map.create(Size(w, h), CV_32FC3);

	const int max_point_pool = (cfg.pose_fast_mode >= 2) ? min(pixel_count, max(65536, cfg.n_poses_to_sample * 4)) : pixel_count;

	float* p2_map_host_out = cfg.pose_fast_mode >= 2 ? NULL : (float*)pts2_map.data;
	float* p3_map_host_out = cfg.pose_fast_mode >= 2 ? NULL : (float*)pts3_map.data;

	if (update_batch_instance) {
		collect_p3p_instances(
			h_flows.data(), h_rigidnesses.data(), (float*)depth.data, (float*)cams[0].K.data, h_Rs.data(), h_ts.data(),
			p2_map_host_out, p3_map_host_out,
			n_flows, w, h, active_idx,
			cfg.rigidness_threshold, cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth, cfg.pose_sample_max_depth, cfg.max_trace_on_flow);
	}
	else if (update_iter_instance) {
		collect_p3p_instances(
			NULL, h_rigidnesses.data(), (float*)depth.data, NULL, h_Rs.data(), h_ts.data(),
			p2_map_host_out, p3_map_host_out,
			n_flows, w, h, active_idx,
			cfg.rigidness_threshold, cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth, cfg.pose_sample_max_depth, cfg.max_trace_on_flow);
	}
	else {
		collect_p3p_instances(
			NULL, NULL, NULL, NULL, h_Rs.data(), h_ts.data(),
			p2_map_host_out, p3_map_host_out,
			n_flows, w, h, active_idx,
			cfg.rigidness_threshold, cfg.rigidness_sum_threshold,
			cfg.pose_sample_min_depth, cfg.pose_sample_max_depth, cfg.max_trace_on_flow);
	}

	if (!cfg.silent)
		cout << "[pose] sampling/instance collection time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << endl;
	time_stamp = chrono::high_resolution_clock::now();

	int n_points = 0;
	if (cfg.pose_fast_mode >= 2) {
		compact_p3p_instances(
			(float*)pts2.data(), (float*)pts3.data(),
			&n_points, max_point_pool, w, h);
	}
	else {
		Point2f* pts2_pt = (Point2f*)pts2_map.data;
		Point3f* pts3_pt = (Point3f*)pts3_map.data;
		for (int i = 0; i < pixel_count && n_points < max_point_pool; i++) {
			if (isfinite(pts2_pt->x + pts2_pt->y + pts3_pt->x + pts3_pt->y + pts3_pt->z)) {
				pts2[n_points] = *pts2_pt;
				pts3[n_points] = *pts3_pt;
				n_points++;
			}
			pts2_pt++;
			pts3_pt++;
		}
	}

	if (!cfg.silent)
		cout << "[pose] dense->sparse compaction time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << endl;
	time_stamp = chrono::high_resolution_clock::now();

	// check if able to have at least one pose
	if (n_points < 4) {
		if (!cfg.silent)
			cout << "[pose] optimize_camera_pose total time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - pose_total_t0).count() / 1e6 << "ms (n_points < 4)." << endl;
		return 0;
	}

	int poses_to_sample = cfg.n_poses_to_sample;
	if (cfg.pose_fast_mode >= 2 && successive_pose) {
		const float pose_density = cams[active_idx].pose_density;
		if (pose_density > 0.7f) {
			poses_to_sample = min(cfg.n_poses_to_sample, 4096);
		}
		else if (pose_density > 0.55f) {
			poses_to_sample = min(cfg.n_poses_to_sample, 6144);
		}
		else if (pose_density > 0.45f) {
			int half_samples = cfg.n_poses_to_sample / 2;
			poses_to_sample = half_samples > 1024 ? half_samples : 1024;
		}
	}
	else if (successive_pose && cams[active_idx].pose_density > 0.45f) {
		int half_samples = cfg.n_poses_to_sample / 2;
		poses_to_sample = half_samples > 1024 ? half_samples : 1024;
	}
	int point_limited = n_points * 2;
	if (point_limited < 512)
		point_limited = 512;
	if (poses_to_sample > point_limited)
		poses_to_sample = point_limited;
	if (poses_to_sample < 128)
		poses_to_sample = 128;
	if (poses_to_sample > cfg.n_poses_to_sample)
		poses_to_sample = cfg.n_poses_to_sample;
	if (!cfg.silent)
		cout << "[pose] compacted points = " << n_points << ", poses_to_sample = " << poses_to_sample << endl;

	Mat poses_pool(poses_to_sample, 6, CV_32F);
	int poses_pool_used = 0;

	if (cfg.cpu_p3p) {


		for (int i = 0; i < poses_to_sample; i++) {
			int i1 = ((float)rand() / (float)RAND_MAX)*n_points;
			int i2 = ((float)rand() / (float)RAND_MAX)*n_points;
			int i3 = ((float)rand() / (float)RAND_MAX)*n_points;
			int i4 = ((float)rand() / (float)RAND_MAX)*n_points;


			if (cfg.lambdatwist) {
				float R_temp[3][3];
				Vec3f rvec_temp, tvec_temp;
				if (lambdatwist_p4p<double, float, 5>(
					(float*)&pts2[i1], (float*)&pts2[i2], (float*)&pts2[i3], (float*)&pts2[i4],
					(float*)&pts3[i1], (float*)&pts3[i2], (float*)&pts3[i3], (float*)&pts3[i4],
					cfg.fx, cfg.fy, cfg.cx, cfg.cy,
					R_temp, tvec_temp.val)) {

					Rodrigues(Matx33f((float*)R_temp), rvec_temp);
					//cout <<"rvec = "<< rvec_temp << endl;
					//cout <<"tvec = "<< tvec_temp << endl;

					if (isfinite(tvec_temp[0] + tvec_temp[1] + tvec_temp[2] + rvec_temp[0] + rvec_temp[1] + rvec_temp[2])) {
						poses_pool.at<Vec3f>(poses_pool_used, 0) = rvec_temp;
						poses_pool.at<Vec3f>(poses_pool_used++, 1) = tvec_temp;
					}

				}
			}
			else {
				Point2f p2s_tmp[4] = { pts2[i1],pts2[i2],pts2[i3],pts2[i4] };
				Point3f p3s_tmp[4] = { pts3[i1],pts3[i2],pts3[i3],pts3[i4] };
				Vec3d rvec_temp, tvec_temp;
				if (solvePnP(_InputArray(p3s_tmp, 4), _InputArray(p2s_tmp, 4),
					cams[active_idx].K, Mat(),
					rvec_temp, tvec_temp, false, 5)) { //5 stands for SOLVEPNP_AP3P, does not work ealier opencv version
					//cout <<"tvec = "<< tvec_temp << endl;
					if (isfinite(tvec_temp[0] + tvec_temp[1] + tvec_temp[2] + rvec_temp[0] + rvec_temp[1] + rvec_temp[2])) {
						poses_pool.at<Vec3f>(poses_pool_used, 0) = rvec_temp;
						poses_pool.at<Vec3f>(poses_pool_used++, 1) = tvec_temp;
					}
				}
			}
		}
	}
	else {
		vector<float> ret_Rs_local(poses_to_sample * 3);
		vector<float> ret_ts_local(poses_to_sample * 3);
		thread_local vector<float> ret_Rs_reuse;
		thread_local vector<float> ret_ts_reuse;
		vector<float>& ret_Rs = cfg.pose_fast_mode > 0 ? ret_Rs_reuse : ret_Rs_local;
		vector<float>& ret_ts = cfg.pose_fast_mode > 0 ? ret_ts_reuse : ret_ts_local;
		if ((int)ret_Rs.size() < poses_to_sample * 3)
			ret_Rs.resize(poses_to_sample * 3);
		if ((int)ret_ts.size() < poses_to_sample * 3)
			ret_ts.resize(poses_to_sample * 3);

		if (cfg.lambdatwist) {
			solve_batch_p3p_lambdatwist_gpu((float*)pts3.data(), (float*)pts2.data(), ret_Rs.data(), ret_ts.data(), (float*)cams[active_idx].K.data, n_points, poses_to_sample);
		}
		else {
			solve_batch_p3p_ap3p_gpu((float*)pts3.data(), (float*)pts2.data(), ret_Rs.data(), ret_ts.data(), (float*)cams[active_idx].K.data, n_points, poses_to_sample);
		}

		for (int i = 0; i < poses_to_sample; i++) {
			if (isfinite(ret_Rs[i * 3 + 0] + ret_Rs[i * 3 + 1] + ret_Rs[i * 3 + 2] + ret_ts[i * 3 + 0] + ret_ts[i * 3 + 1] + ret_ts[i * 3 + 2])) {
				// This is a solved bug, atan2 is more accurate than acos2 in rodrigues implementation
				// Due to GPU accuracy issue, GPU p3p sometimes gives pure zero rotation, this will cause incorrect covar in rg_refine process
				//if (rg_refine && (ret_Rs[i * 3 + 0] == 0 && ret_Rs[i * 3 + 1] == 0 && ret_Rs[i * 3 + 2] == 0))
					//continue;
				poses_pool.at<Vec3f>(poses_pool_used, 0) = Vec3f(ret_Rs[i * 3 + 0], ret_Rs[i * 3 + 1], ret_Rs[i * 3 + 2]);
				poses_pool.at<Vec3f>(poses_pool_used++, 1) = Vec3f(ret_ts[i * 3 + 0], ret_ts[i * 3 + 1], ret_ts[i * 3 + 2]);
			}
		}
	}

	if (!cfg.silent)
		cout << "[pose] p3p computing time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << endl;
	time_stamp = chrono::high_resolution_clock::now();

	if (poses_pool_used == 0) {
		if (!cfg.silent)
			cout << "[pose] optimize_camera_pose total time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - pose_total_t0).count() / 1e6 << "ms (no valid pose samples)." << endl;
		return 0;
	}
	poses_pool = poses_pool.rowRange(0, poses_pool_used);
	cams[active_idx].pose_sample_count = poses_pool_used;

	Mat pose_opm(1, 6, CV_32F);
	Rodrigues(cams[active_idx].R, pose_opm.at<Vec3f>(0));
	pose_opm.at<Vec3f>(1) = cams[active_idx].t.at<Vec3f>(0);


	// scale and do meanshift
	time_stamp = chrono::high_resolution_clock::now();

	poses_pool.colRange(0, 3) *= cfg.meanshift_rvec_scale;
	pose_opm.colRange(0, 3) *= cfg.meanshift_rvec_scale;
	meanshift_gpu((float*)poses_pool.data, cfg.meanshift_kernel_var,
		(float*)pose_opm.data, &cams[active_idx].pose_density, &cams[active_idx].last_used_ms_iters, successive_pose,
		poses_pool_used, 6, cfg.meanshift_epsilon, cfg.meanshift_max_iters, cfg.meanshift_max_init_trials, cfg.meanshift_good_init_confidence);

	if (!cfg.silent)
		cout << "[pose] meanshift time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << endl;


	if (rg_refine) {
		time_stamp = chrono::high_resolution_clock::now();

		cams[active_idx].pose_covar = 0;
		for (int d = 0; d < 6; d++)
			cams[active_idx].pose_covar.at<float>(d, d) = cfg.meanshift_kernel_var;

		cams[active_idx].pose_covar *= (cfg.rg_pose_scaling* cfg.rg_pose_scaling);
		pose_opm *= cfg.rg_pose_scaling;
		poses_pool *= cfg.rg_pose_scaling;

		/*
		int rvec_zero_count = 0;
		for (int i = 0; i < poses_pool_used; i++)
			if (poses_pool.at<float>(i, 0) == 0)
				rvec_zero_count++;
		cout << "rvec_zero_count = " << rvec_zero_count << endl;
		*/


		int fit_rg_ret = fit_robust_gaussian((float*)poses_pool.data, (float*)pose_opm.data, (float*)cams[active_idx].pose_covar.data,
			cfg.rg_trunc_sigma, cfg.rg_covar_reg_lambda, &cams[active_idx].pose_density, &cams[active_idx].last_used_gu_iters, poses_pool_used, 6, cfg.rg_epsilon, cfg.rg_max_iters);

		if (fit_rg_ret == 0) { // cudaSuccess==0 
			cams[active_idx].pose_covar /= (cfg.rg_pose_scaling*cfg.rg_pose_scaling);
			for (int i1 = 0; i1 < 6; i1++) {
				for (int i2 = 0; i2 < 6; i2++) {
					if (i1 < 3 || i2 < 3)
						cams[active_idx].pose_covar.at<float>(i1, i2) /= cfg.meanshift_rvec_scale;
					if (i1 < 3 && i2 < 3)
						cams[active_idx].pose_covar.at<float>(i1, i2) /= cfg.meanshift_rvec_scale;
				}
			}

		}
		else {
			cams[active_idx].pose_covar = 0;
		}

		pose_opm /= cfg.rg_pose_scaling;
		//poses_pool /= cfg.rg_pose_scaling; // this is not used later, no need scale back


		if (!cfg.silent)
			cout << "[pose] gu fit time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6 << "ms." << endl;
	}


	pose_opm.colRange(0, 3) /= cfg.meanshift_rvec_scale;
	//poses_pool.colRange(0, 3) /= cfg.meanshift_rvec_scale; // this is not used later, no need scale back



	time_stamp = chrono::high_resolution_clock::now();

	if (checkRange(pose_opm)) { //check if gpu gives nan...
		// copy back pose
		Rodrigues(pose_opm.at<Vec3f>(0), cams[active_idx].R);
		cams[active_idx].t.at<Vec3f>(0) = pose_opm.at<Vec3f>(1);
		if (!cfg.silent)
			cout << "[pose] optimize_camera_pose total time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - pose_total_t0).count() / 1e6 << "ms." << endl;
		return 1;
	}
	else {
		if (!cfg.silent)
			cout << "[pose] optimize_camera_pose total time = " << chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - pose_total_t0).count() / 1e6 << "ms (pose has NAN)." << endl;
		return 0;
	}

}

void estimate_depth_closed_form(Mat flow, Mat& depth, Camera cam,
	float min_depth, float max_depth) {
	Mat b = cam.K * cam.t;
	Mat KRKinv = cam.K * cam.R * cam.K_inv;

	float b1 = b.at<float>(0), b2 = b.at<float>(1), b3 = b.at<float>(2);
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			Point2f delta = flow.at<Point2f>(y, x);
			Mat P = (Mat_<float>(3, 1) << x, y, 1);
			P = KRKinv * P;
			float w1 = P.at<float>(0), w2 = P.at<float>(1), w3 = P.at<float>(2);
			float a1 = x + delta.x, a2 = y + delta.y;
			float z_nume = (a1 * b3 - b1)*(w1 - a1 * w3) + (a2 * b3 - b2)*(w2 - a2 * w3);
			float z_deno = (w1 - a1 * w3)*(w1 - a1 * w3) + (w2 - a2 * w3)*(w2 - a2 * w3);
			depth.at<float>(y, x) = fminf(fmaxf(z_nume / z_deno, min_depth), max_depth);
		}
	}
}


void estimate_camera_pose_epipolar(Mat flow, Camera& cam,
	Mat mask, int sampling_2d_step) {
	int w = flow.cols, h = flow.rows;
	bool use_external_mask = !mask.empty();

	Mat pts1(w*h, 2, CV_32F);
	Mat pts2(w*h, 2, CV_32F);
	float* pts1_iter = (float*)pts1.data;
	float* pts2_iter = (float*)pts2.data;
	int N_used = 0;
	for (int y = 0; y < h; y += sampling_2d_step) {
		for (int x = 0; x < w; x += sampling_2d_step) {
			if (use_external_mask && mask.at<float>(y, x) < 0.5)
				continue;
			N_used++;
			*pts1_iter++ = x;
			*pts1_iter++ = y;
			*pts2_iter++ = x + flow.at<Vec2f>(y, x)[0];
			*pts2_iter++ = y + flow.at<Vec2f>(y, x)[1];
		}
	}

	pts1 = pts1.rowRange(0, N_used);
	pts2 = pts2.rowRange(0, N_used);
	//cam.F = findFundamentalMat(pts1, pts2, CV_RANSAC);

	if (use_external_mask) {
		Mat pts_mask;
		cam.E = findEssentialMat(pts1, pts2, cam.K, LMEDS, 0.999, 1.0, pts_mask);
		//cam.E = findHomography(pts1, pts2, LMEDS, 3.0, pts_mask);
	}
	else {
		Mat pts_mask;
		cam.E = findEssentialMat(pts1, pts2, cam.K, LMEDS, 0.999, 1.0, pts_mask);
		//cam.E = findHomography(pts1, pts2, LMEDS, 3.0, pts_mask);
	}


	recoverPose(cam.E, pts1, pts2, cam.K, cam.R, cam.t);
	cam.E.convertTo(cam.E, CV_32F);
	cam.R.convertTo(cam.R, CV_32F);
	cam.t.convertTo(cam.t, CV_32F);
	cam.t = cam.R*cam.t;

}



KittiGround estimate_kitti_ground_plane(Mat depth, Rect roi, Mat K,
	int holo_width, float ms_kernel_var) {

	int w = depth.cols, h = depth.rows;

	Mat K_inv = K.inv();

	float* ground_params = new float[roi.width*roi.height * 4]; // h, n1,n2,n3
	float* height_params = new float[roi.width*roi.height]; // h (for evaluate norm scale for height)
	int valid_N = 0;

	Point3f* p3_buffer = new Point3f[(holo_width * 2 + 1)*(holo_width * 2 + 1)];

	for (int y = roi.y; y < roi.y + roi.height; y++) {
		for (int x = roi.x; x < roi.x + roi.width; x++) {

			Point3f mean(0, 0, 0);

			int N = 0;
			for (int ky = -holo_width; ky <= holo_width; ky++) {
				for (int kx = -holo_width; kx <= holo_width; kx++) {
					// fall in image
					if (x + kx >= 0 && x + kx < w && y + ky >= 0 && y + ky < h) {
						Point3f p3(x + kx, y + ky, 1);
						p3 = Point3f(p3.dot(K_inv.at<Point3f>(0)),
							p3.dot(K_inv.at<Point3f>(1)), p3.dot(K_inv.at<Point3f>(2))) * depth.at<float>(y + ky, x + kx);
						p3_buffer[N++] = p3;
						mean += p3;
					}
				}
			}

			mean /= N;
			float cov[6]{ 0,0,0,0,0,0 };
			for (int i = 0; i < N; i++) {
				Point3f p3 = p3_buffer[i];
				cov[0] += (p3.x - mean.x)*(p3.x - mean.x);
				cov[1] += (p3.x - mean.x)*(p3.y - mean.y);
				cov[2] += (p3.x - mean.x)*(p3.z - mean.z);
				cov[3] += (p3.y - mean.y)*(p3.y - mean.y);
				cov[4] += (p3.y - mean.y)*(p3.z - mean.z);
				cov[5] += (p3.z - mean.z)*(p3.z - mean.z);
			}
			Matx33f cov_mat(cov[0], cov[1], cov[2],
				cov[1], cov[3], cov[4],
				cov[2], cov[4], cov[5]);

			Matx31f eval;
			Matx33f evec;
			if (eigen(cov_mat, eval, evec)) {

				Vec3f n = Vec3f(evec.val[6], evec.val[7], evec.val[8]);
				n /= norm(n, NORM_L2);
				Point3f p3(x, y, 1);
				p3 = Point3f(p3.dot(K_inv.at<Point3f>(0)),
					p3.dot(K_inv.at<Point3f>(1)), p3.dot(K_inv.at<Point3f>(2))) * depth.at<float>(y, x);

				float height = n.dot(p3);
				if (!isfinite(height))
					continue;

				if (height > 0)
					n = -n; // make normal vector point to view point
				else
					height = -height; // make height positive
				ground_params[valid_N * 4 + 0] = height;
				ground_params[valid_N * 4 + 1] = n.val[0];
				ground_params[valid_N * 4 + 2] = n.val[1];
				ground_params[valid_N * 4 + 3] = n.val[2];
				height_params[valid_N] = height;
				valid_N++;
			}
		}
	}

	KittiGround ret;

	if (valid_N < 1)
		return ret;

	// normalize with height median
	sort(height_params, height_params + valid_N);
	ret._height_median = height_params[valid_N / 2];
	for (int i = 0; i < valid_N; i++)
		ground_params[i * 4] /= ret._height_median;



	Vec4f mean(1, 0, -1, 0);
	meanshift_gpu(ground_params, ms_kernel_var, mean.val, &ret.confidence, &ret.used_iters, true, valid_N, 4);
	ret.height = mean[0] * ret._height_median;
	ret.normal = Vec3f(mean[1], mean[2], mean[3]);

	delete[] ground_params;
	delete[] height_params;
	delete[] p3_buffer;
	return ret;
}