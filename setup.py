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
from collections import OrderedDict
from setuptools import setup, find_packages
from distutils.extension import Extension
import numpy as np

named_arches = OrderedDict([
    ('Kepler+Tesla', '3.7'),
    ('Kepler', '3.5+PTX'),
    ('Maxwell+Tegra', '5.3'),
    ('Maxwell', '5.0;5.2+PTX'),
    ('Pascal', '6.0;6.1+PTX'),
    ('Volta', '7.0+PTX'),
    ('Turing', '7.5+PTX'),
])

supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                    '7.0', '7.2', '7.5']

valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

# SM52 or SM_52, compute_52 
#   – Quadro M6000, 
#   - GeForce 900, GTX-970, GTX-980, 
#   - GTX Titan X
# SM61 or SM_61, compute_61 
#   – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030,
#   - Titan Xp, Tesla P40, Tesla P4, 
#   - Discrete GPU on the NVIDIA Drive PX2
# SM75 or SM_75, compute_75 
#   – GTX/RTX Turing 
#   – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, 
#   - Titan RTX,

'''
Determine CUDA arch flags to use.
    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

-gencode=arch=compute_52,code=sm_52
-gencode=arch=compute_61,code=sm_61
-gencode=arch=compute_75,code=sm_75
'''

def get_cuda_arch_flags(arch_list='5.2;6.1;7.5'):
    # arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '5.2;6.1;7.5')
    if arch_list is not None:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = arch_list.replace(' ', ';')
        for named_arch, archval in named_arches.items():
            arch_list = arch_list.replace(named_arch, archval)
        arch_list = arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))
        else:
            num = arch[0] + arch[2]
            flags.append('-gencode=arch=compute_{},code=sm_{}'.format(num, num))
            if arch.endswith('+PTX'):
                flags.append('-gencode=arch=compute_{},code=compute_{}'.format(num, num))
    return flags
    # return list(set(flags))

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
                            'nvcc': get_cuda_arch_flags() + [ 
                                     # '-arch=sm_35',
                                     # '--ptxas-options=-v',
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
        # FIXME PYPI version rejects sha
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
    pkgs = ['dcn', 'rfcn']
    for pkg in pkgs:
        version = write_version_py(pkg)
    setup(
        name='-'.join(map(lambda pkg: pkg.upper(), pkgs)),
        version=version,
        author='Farley Lai',
        url='https://github.com/necla-ml/Deformable-ConvNets-py3',
        description=f"Forked Deformable-ConvNets for Python 3",
        long_description=readme(),
        long_description_content_type='text/markdown',
        keywords='computer vision, object detection',
        license='MIT',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Operating System :: POSIX :: Linux',
            'Development Status :: 4 - Beta',
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
        # install_requires=['easydict', 'mxnet-cu101mkl>=1.6.0'],
        install_requires=['easydict'],
        ext_modules=ext_modules,
        cmdclass=dict(build_ext=custom_build_ext),
        zip_safe=False)
