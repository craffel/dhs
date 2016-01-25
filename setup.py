from setuptools import setup
import numpy.distutils
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [Extension(
    "dhs.int_dist", [os.path.join("dhs", "int_dist.pyx")],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    extra_compile_args=['-mpopcnt'])]


setup(
    name='dhs',
    version='0.0.0',
    description=('Learning to convert sequences of feature vectors to '
                 'downsampled sequences of hashes'),
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/dhs',
    packages=['dhs'],
    long_description="""\
    Learning to convert sequences of feature vectors to downsampled sequences
    of hashes.
    """,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords='dtw',
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
        'numba',
        'scipy'
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
