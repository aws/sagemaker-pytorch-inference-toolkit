import unittest
from unittest.mock import patch, MagicMock
import os, subprocess
import sys
import io

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from sagemaker_pytorch_serving_container.serving import configure_logging

class TestLogConfig(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.environ.get')
    @patch('subprocess.run')
    def test_valid_log_level(self, mock_run, mock_env_get, mock_exists):
        mock_exists.return_value = True
        mock_env_get.return_value = '20'
        mock_run.return_value = MagicMock(returncode=0)
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            configure_logging()
            self.assertIn("Logging level set to error", fake_out.getvalue())
        
        mock_run.assert_called_once()

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_invalid_log_level(self, mock_env_get, mock_exists):
        mock_exists.return_value = True
        mock_env_get.return_value = '70'
        
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            configure_logging()
            self.assertIn("Invalid TS_LOG_LEVEL value: 70", fake_err.getvalue())

    @patch('os.path.exists')
    @patch('os.environ.get')
    def test_no_log_level_set(self, mock_env_get, mock_exists):
        mock_exists.return_value = True
        mock_env_get.return_value = None
        
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            configure_logging()
            self.assertIn("TS_LOG_LEVEL not set", fake_out.getvalue())

    @patch('os.path.exists')
    @patch('os.environ.get')
    @patch('subprocess.run')
    def test_subprocess_error(self, mock_run, mock_env_get, mock_exists):
        mock_exists.return_value = True
        mock_env_get.return_value = '20'
        mock_run.side_effect = subprocess.CalledProcessError(1, 'sed')
        
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            configure_logging()
            self.assertIn("Error configuring the logging", fake_err.getvalue())

    @patch('os.path.exists')
    def test_log4j2_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            configure_logging()
            self.assertIn("does not exist", fake_err.getvalue())

if __name__ == '__main__':
    unittest.main()
