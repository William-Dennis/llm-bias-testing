import logging
import subprocess
import time
import atexit
import platform
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


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
                logger.exception("Failed to kill existing ollama process on Windows")
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
                logger.exception("Failed to kill existing ollama process")

    def start(self):
        if self.process is not None:
            return  # already running

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._wait_for_server()
        atexit.register(self.stop)

    def _wait_for_server(self, timeout: int = 30, interval: int = 1):
        url = "http://localhost:11434/api/tags"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    logger.info("Ollama server is ready")
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(interval)
        # Terminate the process and read stderr via communicate()
        stderr_output = ""
        if self.process:
            try:
                _, stderr_output = self.process.communicate(timeout=5)
                stderr_output = stderr_output.decode("utf-8", errors="replace") if stderr_output else ""
            except Exception:
                logger.exception("Failed to read ollama server stderr")
                self.process.kill()
            self.process = None
        if stderr_output:
            logger.error("Ollama server stderr:\n%s", stderr_output)
        raise RuntimeError(f"Ollama server did not start within {timeout} seconds")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
