# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_pytorch_inference',
    version=read('VERSION').strip(),
    description='Open source library for creating PyTorch containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    package_data={'': ["etc/*"]},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-pytorch-inference-toolkit',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],

    # We don't declare our dependency on torch here because we build with
    # different packages for different variants
    install_requires=['numpy==1.24.4', 'retrying==1.3.4', 'sagemaker-inference==1.10.0'],
    extras_require={
        'test': ['boto3==1.28.60', 'coverage==7.3.2', 'docker-compose==1.29.2', 'flake8==6.1.0', 'Flask==3.0.0',
                 'mock==5.1.0', 'pytest==7.4.2', 'pytest-cov==4.1.0', 'pytest-xdist==3.3.1', 'PyYAML==5.4.1',
                 'sagemaker==2.125.0', 'six==1.16.0', 'requests==2.31.0',
                 'requests_mock==1.11.0', 'torch==2.1.0', 'torchvision==0.16.0', 'tox==4.11.3']
    },

    entry_points={
        'console_scripts': 'serve=sagemaker_pytorch_serving_container.serving:main'
    }
)
