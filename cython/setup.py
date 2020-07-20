from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy

series_params = {
                    'language' : 'c++',
                    'include_dirs' : [r'.', r'../..', numpy.get_include()],
                    'library_dirs' : [r'.', r'..'],
                    'libraries' : ['c', 'stdc++'],
                    'extra_objects' : ['../libfaiss.a']
                }

series = Extension('cy_faiss', ['IndexWrapper.pyx'], **series_params)

setup_params =   {
                    'name' : 'cy_faiss',
                    'cmdclass' : {'build_ext' : build_ext},
                    'ext_modules' : [series]
                }

setup(**setup_params)