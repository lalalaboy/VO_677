#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "${0%/*}/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/demo"
ARTIFACT_DIR="${GSVOLDOR_ARTIFACT_DIR:-}"
IMAGE_ARTIFACT_DIR="/opt/gsvoldor-bin"

rm -f "${DEMO_DIR}"/libgpu-kernels.so
rm -f "${DEMO_DIR}"/pyvoldor_full*.so
rm -f "${DEMO_DIR}"/pyDBoW3*.so

cd "${ROOT_DIR}/slam_py/install"
rm -rf build frame-alignment pose-graph voldor pyvoldor_full.cpp
python setup_linux_full.py build_ext -i

mv libgpu-kernels.so "${DEMO_DIR}/"
mv pyvoldor_full*.so "${DEMO_DIR}/"



if compgen -G "${IMAGE_ARTIFACT_DIR}/pyDBoW3*.so" > /dev/null; then
  cp -f "${IMAGE_ARTIFACT_DIR}"/pyDBoW3*.so "${DEMO_DIR}/"
elif [[ -n "${ARTIFACT_DIR}" ]] && compgen -G "${ARTIFACT_DIR}/pyDBoW3*.so" > /dev/null; then
  cp -f "${ARTIFACT_DIR}"/pyDBoW3*.so "${DEMO_DIR}/"
fi


