# docker build --no-cache --progress=plain -t gsvoldor:cuda11.4 .

DOCKER_BUILDKIT=1 docker build --ssh default=$HOME/.ssh/id_ed25519 -t gsvoldor:cuda11.4 .


GPU_DEVICE=${GPU_DEVICE:-1}
NO_VIEWER=${NO_VIEWER:-0}

xhost +local:docker
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e CUDA_VISIBLE_DEVICES=${GPU_DEVICE} \
  -e NO_VIEWER=${NO_VIEWER} \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace/gsvoldor \
  -w /workspace/gsvoldor \
  gsvoldor:cuda11.4 \
  bash -lc "./scripts/build.sh && ./scripts/run.sh"

# docker run --gpus all -it \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $(pwd):/workspace/gsvoldor \
#   -w /workspace/gsvoldor \
#   gsvoldor:cuda11.4 
