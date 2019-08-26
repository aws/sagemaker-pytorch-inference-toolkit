import shlex
import subprocess
import sys
import os.path

from sagemaker_pytorch_serving_container import serving_backup

if not os.path.exists("/opt/ml/input/config"):
    subprocess.call(['python', '/usr/local/bin/deep_learning_container.py', '&>/dev/null', '&'])

if sys.argv[1] == 'serve':
    serving_backup.main()
else:
    subprocess.check_call(shlex.split(' '.join(sys.argv[1:])))

# prevent docker exit
subprocess.call(['tail', '-f', '/dev/null'])