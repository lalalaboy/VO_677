# docker build --no-cache --progress=plain -t gsvoldor:cuda11.4 .

DOCKER_BUILDKIT=1 docker build --ssh default=$HOME/.ssh/id_ed25519 -t gsvoldor:cuda11.4 .


xhost +local:docker
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
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
