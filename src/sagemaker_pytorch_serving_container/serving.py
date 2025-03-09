# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os,sys
import subprocess
from subprocess import CalledProcessError

from retrying import retry
from sagemaker_pytorch_serving_container import torchserve
from sagemaker_pytorch_serving_container import handler_service

HANDLER_SERVICE = handler_service.__file__

## added logging function to configure log4j2 loglevel.
def configure_logging():
    log_levels = {
        '0': 'off',
        '10': 'fatal',
        '20': 'error',
        '30': 'warn',
        '40': 'info',
        '50': 'debug',
        '60': 'trace'
    }

    # Get the directory of the current script
    current_script_path = os.path.abspath(__file__)
    
    # Construct the path to log4j2.xml relative to the script location
    log4j2_path = os.path.join(os.path.dirname(current_script_path), 'etc', 'log4j2.xml')

    print(f"Current script path: {current_script_path}")
    print(f"log4j2.xml path: {log4j2_path}")

    if not os.path.exists(log4j2_path):
        print(f"Error: {log4j2_path} does not exist", file=sys.stderr)
        return

    ts_log_level = os.environ.get('TS_LOG_LEVEL')

    if ts_log_level is not None:
        if ts_log_level in log_levels:
            try:
                log_level = log_levels[ts_log_level]
                subprocess.run(['sed', '-i', f's/info/{log_level}/g', log4j2_path], check=True)
                print(f"Logging level set to {log_level}")
            except subprocess.CalledProcessError as e:
                print(f"Error configuring the logging: {e}", file=sys.stderr)
        else:
            print(f"Invalid TS_LOG_LEVEL value: {ts_log_level}. No changes made to logging configuration.", file=sys.stderr)
    else:
        print("TS_LOG_LEVEL not set. Using default logging configuration.")

def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError)

@retry(stop_max_delay=1000 * 30,
       retry_on_exception=_retry_if_error)
def _start_torchserve():
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    torchserve.start_torchserve(handler_service=HANDLER_SERVICE)

def main():
    configure_logging()
    _start_torchserve()

if __name__ == '__main__':
    main()
