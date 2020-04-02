# Changelog

## v1.3.1 (2020-04-01)

### Bug Fixes and Other Changes

 * upgrade inference-toolkit version

## v1.3.0 (2020-03-18)

### Features

 * install inference toolkit from PyPI.

### Bug Fixes and Other Changes

 * Fixed awscli
 * awscli update
 * Update awscli
 * Add multi-model capability label to 1.4.0 Dockerfile
 * Upgrade the version of sagemaker-inference
 * default model_fn and predict_fn in default handler

### Documentation Changes

 * update README with pytorch eia section

### Testing and Release Infrastructure

 * refactor toolkit tests.

## v1.2.1 (2020-02-24)

### Bug Fixes and Other Changes

 * update: Update PyTorch-EI health check binary version to match ECL
 * Adding PyTorch EI Support
 * undo upgrade mms version and library name
 * remove multi-model label from dockerfiles

## v1.2.0 (2020-02-20)

### Features

 * Remove torch as requirement for sagemaker_pytorch_inference

### Bug Fixes and Other Changes

 * copy all tests to test-toolkit folder.
 * Remove awscli pin to avoid broken requirements

## v1.1.1 (2020-02-17)

### Bug Fixes and Other Changes

 * update: Update license URL

## v1.1.0 (2020-02-13)

### Features

 * Add release to PyPI. Change package name to sagemaker-pytorch-inference.

### Bug Fixes and Other Changes

 * Fix version.
 * Fix py2 test environment.
 * Adding changes for PyTorch 1.4.0 DLC
 * Update artifacts
 * Create __init__.py
 * run local GPU tests for Python 3
 * update: Update buildspec for PyTorch 1.3.1
 * update copyright year in license header
 * fix year in copyright license header
 * update gitignore
 * remove unused test files
 * add flake8 check to PR build
 * upgrade sagemaker to 1.48.0 for test dependencies
 * Updated Pillow version to 6.2.1
 * Add LABEL for port bind to 1.2.0 and 1.3.1 dockerfiles
 * Updated awscli version
 * Added license file
 * Disabling logs in deep_learning_container.py
 * Wheel name typo fix
 * Bump inference-toolkit version and add mme label

### Documentation Changes

 * Fix README.

### Testing and Release Infrastructure

 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v1.0.5 (2019-08-05)

### Bug fixes and other changes

 * upgrade sagemaker-container version
 * unmark 2 deploy tests
 * update p2 restricted regions

## v1.0.6 (2019-06-21)

### Bug fixes and other changes

 * unmark 2 deploy tests

## v1.0.5 (2019-06-20)

### Bug fixes and other changes

 * update p2 restricted regions

## v1.0.4 (2019-06-19)

### Bug fixes and other changes

 * skip tests in gpu instance restricted regions

## v1.0.3 (2019-06-18)

### Bug fixes and other changes

 * modify buildspecs and tox files

## v1.0.2 (2019-06-17)

### Bug fixes and other changes

 * freeze dependency versions

## v1.0.1 (2019-06-13)

### Bug fixes and other changes

 * add buildspec-release file and upgrade cuda version
 * upgrade PyTorch to 1.1
 * disable test_mnist_gpu for py2 for now
 * fix broken line of buildspec
 * prevent hidden errors in buildspec
 * Add AWS CodeBuild buildspec for pull request
 * Bump minimum SageMaker Containers version to 2.4.6 and pin SageMaker Python SDK to 1.18.16
 * fix broken link in README
 * Add timeout to test_mnist_gpu test
 * Use dummy role in tests and update local failure integ test
 * Use the SageMaker Python SDK for local serving integ tests
 * Use the SageMaker Python SDK for local integ distributed training tests
 * Use the SageMaker Python SDK for local integ single-machine training tests
 * Pin fastai version to 1.0.39 in CPU dockerfile
 * Use the SageMaker Python SDK for SageMaker integration tests
 * Add missing rendering dependencies for opencv and a simple test.
 * Add opencv support.
 * Freeze PyYAML version to avoid conflict with Docker Compose
 * Unfreeze numpy version.
 * Freeze TorchVision to 0.2.1
 * Specify region when creating S3 resource in integ tests
 * Read framework version from Python SDK for integ test default
 * Fix unicode display problem in py2 container
 * freeze pip <=18.1, fastai == 1.0.39, numpy <= 1.15.4
 * Add support for fastai (https://github.com/fastai/fastai) library.
 * Remove "requsests" from tests dependencies to avoid regular conflicts with "requests" package from "sagemaker" dependencies.
 * Add support for PyTorch-1.0.
