# source /home/junze/anaconda3/bin/activate vo_gs_baseline
cd ./demo
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64
exec python3.8 demo.py \
  --disp_dir "../data/gt/disp_gt" \
  --flow_dir "../data/input/optical_flow" \
  --img_dir "../data/input/img" \
  --fx 586.27075 \
  --fy 586.27075 \
  --cx 374.04108 \
  --cy 202.26265 \
  --bf 117.254150390625 \
  --mode stereo \
  --enable_mapping \
  --enable_loop_closure "./ORBvoc.bin" \
  --save_poses "../data/result/poses/poses_KITTI.txt" \
  --save_depths "../data/result/depths" \
  # --no_viewer

