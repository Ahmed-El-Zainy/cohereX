"""
vLLM backend for CohereX.

Offloads the ASR step to a vLLM server that exposes the OpenAI-compatible
``/v1/audio/transcriptions`` endpoint. Everything else (VAD, alignment,
diarization) still runs locally, so only ``transcribe_batch`` changes.

Two modes:
- Connect to a vLLM server the user already runs (pass ``base_url``).
- Let CohereX start its own vLLM server and stop it again when the run ends
  (``ManagedVLLMServer``).
"""
import atexit
import io
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import soundfile as sf

from coherex.audio import SAMPLE_RATE
from coherex.log_utils import get_logger

logger = get_logger(__name__)

try:
    import httpx
except ImportError:
    httpx = None

_INSTALL_HINT = 'The vLLM backend needs extra packages. Install them with: pip install "coherex[vllm]"'


def _require_httpx():
    if httpx is None:
        raise ImportError(_INSTALL_HINT)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ManagedVLLMServer:
    """Start a ``vllm serve`` subprocess and shut it down when CohereX is done."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        startup_timeout: float = 600.0,
    ):
        if shutil.which("vllm") is None:
            raise RuntimeError(
                'vLLM is not installed. Install it with: pip install "coherex[vllm]" '
                "(a CUDA GPU is required to serve the model)."
            )
        _require_httpx()
        self.model = model
        self.host = host
        self.port = port or _free_port()
        self.api_key = api_key
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.base_url = f"http://{self.host}:{self.port}"
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> "ManagedVLLMServer":
        cmd = [
            "vllm", "serve", self.model,
            "--trust-remote-code",
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        cmd += self.extra_args

        logger.info(f"Starting vLLM server for '{self.model}' on {self.base_url} ...")
        # Inherit stdout/stderr so the user sees vLLM's own startup logs.
        self._proc = subprocess.Popen(cmd)
        atexit.register(self.stop)
        self._wait_until_ready()
        logger.info(f"vLLM server is ready at {self.base_url}")
        return self

    def _wait_until_ready(self):
        health_url = f"{self.base_url}/health"
        deadline = time.time() + self.startup_timeout
        logger.info("Waiting for the vLLM server to load the model (this can take a few minutes) ...")
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited during startup (exit code {self._proc.returncode}). "
                    "See the vLLM logs above for details."
                )
            try:
                if httpx.get(health_url, timeout=5.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {self.startup_timeout:.0f}s."
        )

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            logger.info("Shutting down the vLLM server started by CohereX ...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not stop gracefully; killing it.")
                self._proc.kill()
                self._proc.wait()
        self._proc = None


class VLLMBackend:
    """Transcribe VAD chunks through a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        supported_languages: Optional[List[str]] = None,
        timeout: float = 120.0,
        server: Optional[ManagedVLLMServer] = None,
        max_workers: int = 8,
    ):
        _require_httpx()
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/audio/transcriptions"
        self.model = model
        self.api_key = api_key
        self.supported_languages = list(supported_languages or [])
        self.max_workers = max_workers
        self._server = server
        self._client = httpx.Client(timeout=timeout)

    def _transcribe_one(self, waveform: np.ndarray, language: str) -> str:
        buffer = io.BytesIO()
        sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self._client.post(
            self.url,
            headers=headers,
            files={"file": ("chunk.wav", buffer, "audio/wav")},
            data={"model": self.model, "language": language},
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def transcribe_batch(self, waveforms: List[np.ndarray], language: str) -> List[str]:
        if not waveforms:
            return []
        with ThreadPoolExecutor(max_workers=min(len(waveforms), self.max_workers)) as pool:
            return list(pool.map(lambda wave: self._transcribe_one(wave, language), waveforms))

    def shutdown(self):
        self._client.close()
        if self._server is not None:
            self._server.stop()
            self._server = None
