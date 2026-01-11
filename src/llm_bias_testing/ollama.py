import subprocess
import time
import atexit
import platform
import os
import signal


class OllamaServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        kill_existing: bool = False,
    ):
        self.host = host
        self.port = port
        self.process = None

        if kill_existing:
            self._kill_existing_on_port()

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _kill_existing_on_port(self):
        """
        Intentionally conservative:
        do NOT kill all ollama.exe processes anymore.
        """
        if platform.system() == "Windows":
            # Best-effort: kill by port using netstat
            try:
                result = subprocess.check_output(
                    f'netstat -ano | findstr :{self.port}',
                    shell=True,
                    text=True,
                )
                for line in result.splitlines():
                    pid = int(line.strip().split()[-1])
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except Exception:
                pass
        else:
            try:
                subprocess.run(
                    ["lsof", "-ti", f"tcp:{self.port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    def start(self):
        if self.process is not None:
            return

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(2)
        atexit.register(self.stop)

    def stop(self):
        if self.process is None:
            return

        try:
            if platform.system() == "Windows":
                self.process.terminate()
            else:
                os.kill(self.process.pid, signal.SIGTERM)
        except Exception:
            pass

        self.process.wait(timeout=5)
        self.process = None
