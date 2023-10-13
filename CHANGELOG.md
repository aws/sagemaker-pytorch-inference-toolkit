# Changelog

## v2.0.18 (2023-10-10)

### Bug Fixes and Other Changes

 * Fix integration tests and update Python versions

## v2.0.17 (2023-08-07)

### Bug Fixes and Other Changes

 * remove unused file due to security

## v2.0.16 (2023-07-24)

### Bug Fixes and Other Changes

 * reuse sagemaker-inference's requirements.txt installation logic

## v2.0.15 (2023-05-29)

### Bug Fixes and Other Changes

 * Enable telemetry.

## v2.0.14 (2023-03-22)

### Bug Fixes and Other Changes

 * Modify log4j2.xml to remove dependency on javascript.

## v2.0.13 (2023-03-20)

### Bug Fixes and Other Changes

 * add vmargs=-XX:-UseContainerSupport in config

## v2.0.12 (2022-11-29)

### Bug Fixes and Other Changes

 * Update PyTorch Inference toolkit to log telemetry metrics

## v2.0.11 (2022-11-07)

## v2.0.10 (2022-04-07)

### Bug Fixes and Other Changes

 * pass model directory as input to torchserve

## v2.0.9 (2022-04-04)

### Bug Fixes and Other Changes

 * Update CI to use PyTorch 1.10

## v2.0.8 (2022-01-13)

### Bug Fixes and Other Changes

 * log4j migration from 1 to 2. Replace properties file with xml.

## v2.0.7 (2021-10-26)

### Bug Fixes and Other Changes

 * Enable default model fn for cpu and gpu

## v2.0.6 (2021-10-04)

### Bug Fixes and Other Changes

 * Env variable support for batch inference

## v2.0.5 (2021-03-17)

### Bug Fixes and Other Changes

 * Update TS Archiver to v0.3.1 / Integ tests

## v2.0.4 (2020-09-03)

### Bug Fixes and Other Changes

 * construct correct code path and correctly tar up contents

## v2.0.3 (2020-08-24)

### Bug Fixes and Other Changes

 * Add MMS Config for TS

## v2.0.2 (2020-08-12)

### Bug Fixes and Other Changes

 * Add support for Accept headers with multiple MIME types

## v2.0.1 (2020-08-06)

### Bug Fixes and Other Changes

 * Pin sagemaker dependency version to before version 2

## v2.0.0 (2020-08-03)

### Breaking Changes

 * Change Model server to Torchserve for PyTorch Inference

## v1.5.1.post1 (2020-06-25)

### Testing and Release Infrastructure

 * add issue templates

## v1.5.1.post0 (2020-06-18)

### Testing and Release Infrastructure

 * Make docker folder read only, remove unused tests, update reponame, remove incorrect documentation.

## v1.5.1 (2020-06-16)

### Bug Fixes and Other Changes

 * upgrade dependency versions

## v1.5.0 (2020-05-12)

### Features

 * Python 3.7 support

## v1.4.4 (2020-05-04)

### Bug Fixes and Other Changes

 * Add code_dir to sys.path in HandlerService for MME

## v1.4.3.post0 (2020-04-30)

### Testing and Release Infrastructure

 * use tox in buildspec

## v1.4.3 (2020-04-28)

### Bug Fixes and Other Changes

 * Adding Dockerfile for PT1.5

## v1.4.2 (2020-04-20)

### Bug Fixes and Other Changes

 * Install awscli in 1.3.1 eia from pypi instead of conda

## v1.4.1 (2020-04-16)

### Bug Fixes and Other Changes

 * load EI model to CPU by default in model_fn
 * Update package versions to fix safety check vulnerabilities
 * Change miniconda installation in dockerfiles

## v1.4.0 (2020-04-06)

### Features

 * Remove unnecessary dependencies.

### Bug Fixes and Other Changes

 * upgrade awscli to fix pip check issue
 * upgrade pillow at end of build

## v1.3.2 (2020-04-02)

### Bug Fixes and Other Changes

 * upgrade pillow etc. to fix safety issues in 1.4.0 dockerfiles

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
