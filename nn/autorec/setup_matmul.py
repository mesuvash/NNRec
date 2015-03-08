from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_module = Extension(
    "cython_matmul",
    ["cython_matmul.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
 	include_dirs=[np.get_include()]   	
)

setup(
    name='cython helpers',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[np.get_include()]
)
