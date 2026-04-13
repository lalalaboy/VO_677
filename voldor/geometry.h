#pragma once
#include "utils.h"
#include "config.h"

struct PoseOptimizeTiming {
	float sampling_collection_ms = 0.0f;
	float p3p_computing_ms = 0.0f;
	float meanshift_ms = 0.0f;
	float gu_fit_ms = 0.0f;
};

int optimize_camera_pose(vector<Mat> flows, vector<Mat> rigidnesses, 
	Mat depth, vector<Camera>& cams,
	int n_flows, int active_idx, bool successive_pose,
	bool rg_refine,
	bool update_batch_instance, bool update_iter_instance,
	Config cfg,
	PoseOptimizeTiming* timing = nullptr);

void estimate_depth_closed_form(Mat flow, Mat& depth, Camera cam,
	float min_depth = 1e-2f, float max_depth = 1e10f);


void estimate_camera_pose_epipolar(Mat flow, Camera& cam, 
	Mat mask = Mat(), int sampling_2d_step = 4);



KittiGround estimate_kitti_ground_plane(Mat depth, Rect roi, Mat K,
	int holo_width = 4,
	float ms_kernel_var = 0.01f);
