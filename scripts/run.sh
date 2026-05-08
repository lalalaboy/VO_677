# source /home/junze/anaconda3/bin/activate vo_gs_baseline
cd ./demo
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64
export GLOG_minloglevel=2
export GLOG_logtostderr=1
NO_VIEWER=${NO_VIEWER:-0}
VIEWER_ARGS=""
if [ "$NO_VIEWER" = "1" ]; then
  VIEWER_ARGS="--no_viewer"
fi
exec python3.8 demo.py \
  --disp_dir "../data/input/tartanair/Oldtown/P000/disp_left" \
  --flow_dir "../data/input/tartanair/Oldtown/P000/flow_oldtown_p000" \
  --img_dir "../data/input/tartanair/Oldtown/P000/image_left" \
  --fx 320 \
  --fy 320 \
  --cx 320 \
  --cy 240 \
  --bf 80 \
  --resize 0.5 \
  --abs_resize 0.5 \
  --no_viewer \
  --mode stereo \
  ${VIEWER_ARGS} \
  --save_poses "../data/result/tartanair/oldtown/P000/poses/poses_KITTI.txt" \
  --save_depths "../data/result/tartanair/oldtown/P000/depths"

#--enable_mapping \
#--enable_loop_closure "./ORBvoc.bin" \
#--no_viewer \

