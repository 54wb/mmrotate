import os
import numpy as np
from setuptools import setup, Extension
from Cython.Distutils import build_ext

def make_cpp_ext(name: str, module: str, sources: list, include_dirs: list, extra_include_dirs: list = None): 
    dir_path = os.path.dirname(os.path.abspath(__file__)) # real path 会解析软链接
    include_dirs = [os.path.join(dir_path, *module.split('.'), p) for p in include_dirs]
    if extra_include_dirs is not None:
        include_dirs.extend(extra_include_dirs)
    return Extension(
        name = f'{module}.{name}',
        sources = [os.path.join(*module.split('.'), p) for p  in sources],
        include_dirs = include_dirs,
        language="c++"
    )

setup(
    name="utils",
    version="1.0.0",
    ext_modules=[
        make_cpp_ext(
            name="obb_nms",
            module="utils.obb_nms",
            sources=["obb_nms.pyx", "m_obb_nms.cc"],
            include_dirs=[],
            extra_include_dirs=[np.get_include()]
        )
    ],
    cmdclass={'build_ext': build_ext},
)
