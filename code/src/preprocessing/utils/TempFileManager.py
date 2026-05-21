import os
import shutil
import atexit
import signal

class TempFileManager:
    """
    A class to manage temporary files and directories created during the processing.
    """
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.cleanup_signal)
        signal.signal(signal.SIGTERM, self.cleanup_signal)
        
    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
    def cleanup_signal(self, signum):
        self.cleanup()
        raise SystemExit(f"Received signal {signum}. Exiting...")