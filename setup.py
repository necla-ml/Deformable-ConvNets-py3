# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# py-faster-rcnn
# Copyright (c) 2016 by Contributors
# Licence under The MIT License
# py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup, find_packages
from distutils.extension import Extension
import numpy as np

def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

CUDA = locate_cuda()
def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
from Cython.Distutils import build_ext
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "dcn.bbox.cpu",
        ["dcn/bbox/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        "dcn.nms.cpu",
        ["dcn/nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension('dcn.nms.cuda',
        ['dcn/nms/nms_kernel.cu', 'dcn/nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with
        # gcc the implementation of this trick is in customize_compiler() below
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_35',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include']]
    ),
]

from shutil import *
import subprocess
def sh(*args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', exitcode=False, shell=True, check=False):
    if len(args) == 1:
        args = args[0]

    if args:   
        proc = subprocess.run(args, stdout=stdout, stderr=stderr, shell=shell, check=check)
        output = proc.stdout.decode('utf-8').strip()
    else:
        exitcode = 0
        output = None
    return (proc.returncode, output) if exitcode else output

def write_version_py(path, major=None, minor=None, patch=None, suffix='', sha='Unknown'):
    if major is None or minor is None or patch is None:
        major, minor, patch = sh("git describe --abbrev=0 --tags")[1:].split('.')
        sha = sh("git rev-parse HEAD")
        print(f"Build version {major}.{minor}.{patch}-{sha}")

    from pathlib import Path
    path = Path(path).resolve()
    pkg = path.name
    PKG = pkg.upper()
    version = f'{major}.{minor}.{patch}{suffix}'
    if os.getenv(f'{PKG}_BUILD_VERSION'):
        assert os.getenv(f'{PKG}_BUILD_NUMBER') is not None
        build_number = int(os.getenv(f'{PKG}_BUILD_NUMBER'))
        version = os.getenv(f'{PKG}_BUILD_VERSION')
        if build_number > 1:
            version += '.post' + str(build_number)
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    import time
    content = f"""# GENERATED VERSION FILE
# TIME: {time.asctime()}
__version__ = {repr(version)}
git_version = {repr(sha)}
"""

    with open(path / 'version.py', 'w') as f:
        f.write(content)
    
    return version

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    pkg = 'dcn'
    version = write_version_py(pkg)
    setup(
        name='DCN',
        version=version,
        author='Farley Lai',
        url='https://github.com/necla-ml/Deformable-ConvNets-py3',
        description=f"Forked Deformable-ConvNets for Python 3",
        long_description=readme(),
        keywords='computer vision, object detection',
        license='MIT',
        classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: macOS/Ubuntu 16.04+',
            'Development Status :: 1 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        packages=find_packages(exclude=['experiments', 'faster_rcnn', 'fpn', 'model']),
        package_data=dict(
            rfcn=['cfgs/rfcn_coco_demo*.yaml']
        ),
        setup_requires=['cython'],
        install_requires=['easydict', 'mxnet-cu101mkl>=1.6.0'],
        ext_modules=ext_modules,
        cmdclass=dict(build_ext=custom_build_ext),
        zip_safe=False)
