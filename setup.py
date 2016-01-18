from setuptools import setup

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
)
