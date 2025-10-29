# pose_estimation_user_server_executor.py

import os
from pathlib import Path

from PyServerManager.async_server.async_pickle_client import AsyncPickleClient
from PyServerManager.templates.base_user_server_executor import BaseUserServerExecutor

# Let's assume your PoseEstimationServerExecutor is in a file, e.g. pose_estimation_server_executor.py,
# and you want to run that script. Make sure the path is correct below.
EXECUTOR_SCRIPT = str((Path(__file__).parent.parent / "executors" / "server_pose_executor.py").resolve())


class PoseEstimationUserServerExecutor(BaseUserServerExecutor):
    """
    Inherit from BaseUserServerExecutor, but override the default script path
    to point to our PoseEstimationServerExecutor.
    """

    EXECUTOR_SCRIPT_PATH = EXECUTOR_SCRIPT

    def __init__(self, python_exe=None, activation_script=None, env_vars=None, logger=None):
        super().__init__(
            python_exe=python_exe,
            activation_script=activation_script,
            env_vars=env_vars,
            logger=logger
        )

    def run_pose_server(self, host="localhost", port=5050, open_new_terminal=True, **extra_args):
        """
        Similar to run_server(), but we allow extra_args to be passed for model config, etc.
        E.g. config_dir, model_dir, device, bbox_thr, nms_thr
        """
        self.host = host
        self.port = port
        if host is None:
            self.logger.error(f"Host cannot be None.")
            return
        if port is None:
            self.logger.error(f"Port cannot be None.")
            return
            # self.port = AsyncPickleClient.find_available_port()
            # self.host = host if host else "localhost"
        self.logger.info(f"Starting PoseEstimationServerExecutor on {host}:{port}...")

        # Build the dictionary to pass as script arguments
        args_dict = {
            "host": self.host,
            "port": self.port,
            # Any additional stuff you might want to pass
            **extra_args
        }

        # Call our 'execute()' with encode_args=True
        # so it passes them as base64-encoded JSON to your server executor.
        thread = self.execute(
            args_dict=args_dict,
            encode_args=True,
            open_new_terminal=open_new_terminal,
        )

        # We'll do an async attempt
        # self.client.connect_client()
        self.connect_client(start_sleep=10, max_retries=10)
        self.logger.info(
            f"PoseEstimationServerExecutor started on {host}:{port} "
            f"with PID {thread}."
        )
        return thread


if __name__ == '__main__':

    # Directories for configs and checkpoints
    base_dir = os.path.dirname(os.path.dirname(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    config_dir = os.path.join(base_dir, 'configs')
    model_dir = os.path.join(parent_dir, 'checkpoints')
    port = AsyncPickleClient.find_available_port()
    host = '127.0.0.1'
    pose_estimator = PoseEstimationUserServerExecutor()
    pose_estimator.run_pose_server(
        host=host,
        port=port,
        management_port=port + 1,
        open_new_terminal=True,
        config_dir=config_dir,
        model_dir=model_dir,
        device="cuda:0"
    )
    while True:
        pass
