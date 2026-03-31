from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy
import subprocess
import shutil

# RTX 5000 Ada -> SM 89
nvcc_machine_code = '-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89'

gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

def resolve_opencv_pkg():
    for pkg in ('opencv4', 'opencv'):
        try:
            libs = subprocess.check_output(['pkg-config', '--libs', pkg], text=True).split()
            cflags = subprocess.check_output(['pkg-config', '--cflags', pkg], text=True).split()
            return libs, cflags
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError('pkg-config cannot find either opencv or opencv4.')

opencv_link_flags, opencv_cflags = resolve_opencv_pkg()
opencv_libs = [flag[2:] for flag in opencv_link_flags if flag.startswith('-l')]
opencv_lib_dirs = [flag[2:] for flag in opencv_link_flags if flag.startswith('-L')]
opencv_include_dirs = [flag[2:] for flag in opencv_cflags if flag.startswith('-I')]
opencv_nvcc_include_flags = ' '.join(f'-I{inc}' for inc in opencv_include_dirs)
ceres_lib_dirs = ['/home/junze/ceres-solver/build_vo_baseline/lib']
ceres_include_dirs = [
    '/home/junze/ceres-solver/include',
    '/home/junze/ceres-solver/build_vo_baseline/include',
]

gpu_kernel_build_cmd = (
    f'/usr/local/cuda/bin/nvcc --compiler-options "-shared -fPIC" '
    f'{opencv_nvcc_include_flags} '
    f'{gpu_sources_cpp} {gpu_sources_cu} -lib -o libgpu-kernels.so -O3 {nvcc_machine_code}'
)
if os.system(gpu_kernel_build_cmd) != 0:
    raise SystemExit('Failed to build CUDA gpu kernels with nvcc.')


ext = Extension('pyvoldor_full',
    sources = ['pyvoldor_full.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../frame-alignment/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../pose-graph/*.cpp') if 'main.cpp' not in x],
    language = 'c++',
    library_dirs = ['.'] + ceres_lib_dirs + ['/usr/local/lib', '/usr/local/cuda/lib64'] + opencv_lib_dirs,
    libraries = ['gpu-kernels', 'cudart', 'ceres', 'glog', \
            'amd','btf','camd','ccolamd','cholmod','colamd','cxsparse',\
            'graphblas','klu','ldl','rbio','spqr','umfpack', 'lapack', 'blas', 'gcc'] + opencv_libs,
    include_dirs = [numpy.get_include(), '/usr/include/eigen3'] + opencv_include_dirs + ceres_include_dirs,
    runtime_library_dirs = ceres_lib_dirs + ['/usr/local/cuda/lib64'],
    extra_link_args = [f'-Wl,-rpath,{ceres_lib_dirs[0]}']
)

setup(
    name='pyvoldor_full',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)
