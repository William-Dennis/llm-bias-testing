import subprocess
import time
import atexit
import platform


class OllamaServer:
    def __init__(self, kill_existing: bool = True):
        self.process = None
        if kill_existing:
            self._kill_existing_ollama()

    def _kill_existing_ollama(self):
        # Windows
        if platform.system() == "Windows":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass
        # macOS / Linux
        else:
            try:
                subprocess.run(
                    ["pkill", "-f", "ollama serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    def start(self):
        if self.process is not None:
            return  # already running

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)  # wait for server to start
        atexit.register(self.stop)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
