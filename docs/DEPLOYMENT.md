# Deploying CohereX on a CPU-only server

This documents a real deployment of CohereX's vLLM backend to a small CPU-only VM
(2 vCPU, 8GB RAM, no GPU, Ubuntu 24.04, `x86_64` with AVX-512). It's here so the
setup is reproducible on another box, or debuggable if the running server acts up.

> A GPU server is strongly preferred (see the main [README](../README.md)).
> This path only exists because the target box has no GPU. Expect CPU inference
> to be well below real-time throughput.

## 1. Confirm the box can run vLLM's CPU backend

vLLM's CPU backend requires AVX-512:

```bash
grep -o "avx512[a-z_]*" /proc/cpuinfo | sort -u
```

If that prints nothing, stop — the CPU backend isn't viable there.

## 2. System packages and toolchain

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-dev python3-pip ffmpeg git \
  gcc-12 g++-12 cmake ninja-build build-essential libnuma-dev pkg-config
```

(On Ubuntu 24.04, `gcc-12`/`g++-12` may not exist in the repos — gcc/g++ 13 works fine too.)

## 3. Build vLLM from source with the CPU target

The published `vllm` PyPI wheel is GPU-oriented. For CPU you build it yourself,
following vLLM's own [CPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/):

```bash
python3 -m venv /opt/coherex-venv
source /opt/coherex-venv/bin/activate
pip install --upgrade pip
pip install wheel packaging ninja "setuptools>=49.4.0" numpy

git clone --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-src
cd /opt/vllm-src

pip install -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

export CC=gcc CXX=g++
export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS=2   # match nproc; more jobs on a low-RAM box risks OOM during compile
pip install -e . --no-build-isolation
```

This compiles oneDNN (Intel's CPU math library) plus vLLM's own CPU kernels.
**On a 2-vCPU box this took roughly an hour.** Watch memory during the build —
individual C++ translation units (especially AVX-512 kernel files) can spike to
2–3GB RSS each; with `MAX_JOBS=2` that's the ceiling on an 8GB box. If you see
swap climbing hard or an `Out of memory: Killed process` in `dmesg`, drop
`MAX_JOBS=1` and resume.

⚠️ **Pitfall — don't let a later `pip install` silently replace this build.**
A shallow git clone has no tags, so `pip` reports the built package as version
`0.1.dev1+<hash>.cpu`. Any later `pip install` that pulls in a `vllm>=X.Y.Z`
constraint (for example `pip install "coherex[vllm]"` or `coherex[all]`) will
see `0.1.dev1 < X.Y.Z`, conclude the requirement isn't satisfied, and silently
**uninstall the CPU build and pull the GPU-oriented PyPI wheel instead** — along
with a large pile of CUDA-only dependencies. This actually happened during this
deployment and had to be fixed by re-running the build. To avoid it:

- Install CohereX **before** or **without** re-triggering vLLM's own dependency
  resolution — e.g. `pip install coherex --no-deps` plus its non-vLLM extras,
  since vLLM and `httpx` are already satisfied once you've built vLLM yourself.
- Or rebuild with `pip install -e . --no-build-isolation --no-deps` after
  uninstalling the wrong wheel, which is what was done here — it's a full
  recompile again (the temp CMake build directory isn't preserved across
  `pip uninstall`), so it's cheaper to just avoid triggering the swap.

## 4. Install CohereX

Reuse the CPU-only torch you already have — don't let plain `pip install torch`
pull the CUDA build:

```bash
pip install coherex[all] --extra-index-url https://download.pytorch.org/whl/cpu --no-deps
pip install <langid/pyannote/etc. deps only if pip's dependency resolver hasn't already satisfied them>
```

(In this deployment, `coherex[all]` was run once *before* discovering the vLLM
pitfall above, which is what triggered it. If you build vLLM first and then
install CohereX with `--no-deps` — relying on the fact that vLLM's own install
already pulled in overlapping deps like `httpx`, `transformers`, `torch` — you
avoid the problem entirely.)

## 5. Hugging Face auth

The ASR model is gated. Accept the model's terms on its Hugging Face page, then,
as the same user that will run the server (root, in this deployment):

```bash
source /opt/coherex-venv/bin/activate
hf auth login
```

## 6. Extra swap (needed on an 8GB box)

Loading a ~4GB checkpoint plus vLLM's own overhead comes close to exhausting
8GB of RAM even before any inference happens; the compilation warmup step
(§8) pushes it over. Add a swapfile as headroom — this doesn't make things
fast, it just turns an OOM-kill into "slow but survives":

```bash
fallocate -l 8G /swapfile-coherex
chmod 600 /swapfile-coherex
mkswap /swapfile-coherex
swapon /swapfile-coherex
echo "/swapfile-coherex none swap sw 0 0" >> /etc/fstab   # persist across reboots
```

## 7. Run vLLM as a systemd service

```ini
# /etc/systemd/system/coherex-vllm.service
[Unit]
Description=CohereX vLLM CPU inference server
After=network.target

[Service]
Type=simple
User=root
Environment=VLLM_CPU_OMP_THREADS_BIND=0-1
ExecStart=/opt/coherex-venv/bin/vllm serve CohereLabs/cohere-transcribe-arabic-07-2026 --trust-remote-code --host 0.0.0.0 --gpu-memory-utilization 0.65 --enforce-eager --port 8000 --api-key <VLLM_API_KEY>
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable coherex-vllm
systemctl start coherex-vllm       # only after `hf auth login` has succeeded
journalctl -u coherex-vllm -f      # watch model load; first run downloads several GB
```

The API key is bound to `0.0.0.0` (reachable from the internet) because the
use case here is calling it from arbitrary projects/machines, not just
localhost. Treat that key as a credential — **never commit the real value to
this repo**; keep it in a secrets manager or a gitignored local file.

`VLLM_CPU_OMP_THREADS_BIND=0-1` pins OpenMP threads to the 2 available cores;
adjust the range to match `nproc` on a different box. See §8 below for why
`--gpu-memory-utilization` and `--enforce-eager` are both required here, not
optional tuning.

**Not yet done:** this box has no firewall configured, so besides SSH (22) and
the vLLM port (8000), whatever else is listening is also reachable from the
internet. Consider `ufw allow 22 && ufw allow 8000 && ufw enable` (test the SSH
rule sticks before you enable it — a mistake here can lock you out).

## 8. Runtime crashes hit on first start, and their fixes

The build succeeding and the service starting are not the same thing. On this
box the service crash-looped through three distinct failures before it served
a single request — all three are already fixed in this checkout / this doc's
systemd unit, but are documented here because the errors are cryptic and any
similar low-RAM CPU deployment will likely hit at least the last two again.

### 8a. `ModuleNotFoundError: No module named 'torchaudio'` / `OSError` loading `_torchaudio.abi3.so`

torchaudio's PyPI releases stopped at 2.11.0, which is ABI-incompatible with
newer torch builds (here, torch 2.13.0+cpu required by vLLM) — importing
`torchaudio` at all raises `OSError` on such a build, and there is no newer
release to upgrade to.

Two places needed torchaudio and neither actually needs its compiled
extension:

- `coherex/alignment.py` imported it unconditionally at module level just to
  check `model_name in torchaudio.pipelines.__all__` for the (rarely used)
  torchaudio-native alignment models (`en/fr/de/es/it` only, and only when no
  `--align_model` is given). Fixed by making the import lazy and falling back
  to `torchaudio = None` on any import failure — see the diff in
  `coherex/alignment.py`. Arabic (the default `--language` here) never hits
  this path at all, since its default alignment model is a HuggingFace
  wav2vec2 model, not a torchaudio pipeline.
- vLLM's own `vllm/transformers_utils/processors/cohere_asr.py` (part of
  vLLM's Cohere-ASR model support, not CohereX) does
  `from torchaudio.functional import melscale_fbanks` to build a mel
  filterbank matrix once at model-construction time. `melscale_fbanks` is
  pure torch/math — no compiled extension involved — so it was ported
  verbatim (helpers included) into a new file,
  `vllm/transformers_utils/processors/_mel_fbanks.py`, in the server's vLLM
  checkout at `/opt/vllm-src`, and the import in `cohere_asr.py` was
  redirected to it. This is a patch to the local vLLM source clone, not to
  CohereX — if you rebuild vLLM from a fresh clone, reapply it (or just
  uninstall torchaudio and see if this file still exists in whatever vLLM
  version you're on).

With both fixed, `pip uninstall -y torchaudio` on the server so
`transformers`'s own `is_torchaudio_available()` check correctly treats it as
absent and skips it too (it does presence-only detection, not "does it
actually import").

### 8b. `ValueError: Available memory ... is less than desired CPU memory utilization`

On vLLM's CPU backend, `--gpu-memory-utilization` (yes, that's still the flag
name) is **not** "how much extra to reserve for KV cache" — it's the total
RAM ceiling for the whole worker process: weights + activations + KV cache
combined, expressed as a fraction of total system RAM. The default (0.9) is
sane for a GPU where model weights live in VRAM and this flag really is just
about the GPU's own headroom; on CPU it collides with the same RAM the
weights themselves need.

This model's checkpoint is 3.85GiB — roughly half of this box's 7.76GiB
reported RAM — so the fraction has to comfortably exceed that, not undercut
it. `0.1`, `0.3` were both tried here and both failed (the requested ceiling
was smaller than what the weights alone needed, mid-load, giving negative
"available for KV cache"). `0.65` (~5GiB ceiling) is what worked. If you
change the model or the box's RAM, recompute: you need `models_weight_size_GB / total_RAM_GB` as a floor, plus real headroom above it for KV cache and
framework overhead — don't just copy `0.65`.

### 8c. OOM-killed during "warming up model for the compilation"

Even with the memory ceiling fixed, the worker was OOM-killed by the kernel
(not by vLLM's own check) during torch-compile's warmup/encoder-cache
profiling step — a memory spike unrelated to the steady-state KV cache
budget. `--enforce-eager` skips ahead-of-time `torch.compile`/inductor
compilation (and its warmup), trading a little inference speed for a much
lower and more predictable peak-memory startup path. Combined with the swap
in §6 as a safety margin, this is what got the service to actually reach
`Application startup complete.`

## 9. Testing the deployment

> For a repeatable, scriptable version of everything in this section —
> `.env`-based config, a live progress bar, saved JSON/text output per file —
> use [`main.py`](../main.py), documented in [TESTING.md](TESTING.md).

Once `journalctl -u coherex-vllm -f` shows `Application startup complete.`
and routes including `/v1/audio/transcriptions`:

```bash
curl -s -o /dev/null -w "http_status=%{http_code}\n" http://<SERVER_IP>:8000/health
# http_status=200
```

### Real API call

```bash
curl http://<SERVER_IP>:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer <VLLM_API_KEY>" \
  -F file=@samples/saudi_business_03min.mp3 \
  -F model=CohereLabs/cohere-transcribe-arabic-07-2026 \
  -F language=ar
```

### Full pipeline through the CLI

This exercises the real path end to end: local VAD (silero, per the new
default) chunks the audio, each chunk is sent to the remote vLLM server for
ASR, then alignment and subtitle formatting run locally again:

```bash
coherex "samples/saudi_business_03min.mp3" \
  --backend vllm --vllm_url http://<SERVER_IP>:8000 --vllm_api_key <VLLM_API_KEY>
```

With the CLI defaults (§ below) this needs no other flags — model, language,
VAD method, and output directory (`out-ar/`) are all already correct for this
sample.

**Client concurrency had to be tuned down for this box.** `coherex/vllm_backend.py`'s `VLLMBackend` defaulted to 8 concurrent chunk requests
and a 120s per-request timeout — reasonable against a real GPU server, but
against 2 vCPUs it meant 8-way contention and every request blowing past the
timeout before finishing. Defaults were changed to `max_workers=2`,
`timeout=300.0`. If you deploy to a beefier server, these can go back up
(there's no CLI flag for them yet — edit the `VLLMBackend.__init__` defaults,
or construct it yourself via the Python API).

### What an actual run looked like on this box

Transcribing `samples/saudi_business_03min.mp3` (≈3 minutes of Arabic audio,
7 VAD chunks):

- Each `/v1/audio/transcriptions` call took roughly 1–2.5 minutes; total
  transcription phase was about 7 minutes with 2 concurrent workers.
- RAM sat at ~7GB/7.8GB used with 2–3GB of swap in active use throughout —
  slow, but stable, no further OOM kills once §8's fixes were applied.
- Local alignment (wav2vec2, `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`,
  downloaded on first use) added under a minute on the client machine.
- Output: coherent, correctly-punctuated Arabic transcript of a PIF
  (Public Investment Fund) governance/investment-committee discussion —
  a clear quality improvement over an earlier, garbled sample transcript
  that predates this deployment's fixes.

**Concurrency ceiling:** this box can reliably do about one transcription job
at a time. It is not sized for concurrent users — see §8b/8c and the sizing
notes below before pointing production traffic at it.

## Sizing notes

- Build peak memory: individual compile units reached ~2.5–3.6GB RSS; total
  system usage peaked around 6GB of 8GB with `MAX_JOBS=2`. Don't raise
  `MAX_JOBS` above `nproc` on a box this small.
- Disk: the vLLM source + build artifacts + full dependency set used about
  19GB. Budget for that plus model weights (downloaded on first server start,
  ~4GB for the Arabic model).
- Runtime memory: expect ~7GB RAM + 2-3GB swap in steady use serving one
  request at a time (see §9). This is not headroom for concurrent requests.
- Expect CPU-only inference to run well under real-time (single chunks took
  1-2.5 minutes each here). If throughput or concurrency matters, move this
  same setup to a GPU box — skip the CPU-specific build steps above, drop
  `--enforce-eager`/`--gpu-memory-utilization`/the swapfile, and just
  `pip install vllm` normally.

---

# CohereX CLI defaults (this fork)

The CLI defaults in this checkout were changed from upstream to match the
primary use case (Arabic transcription with tighter subtitle formatting):

| Flag                      | New default                                     | Upstream default                         | Where                                                                                    |
| ------------------------- | ----------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| `--model`               | `CohereLabs/cohere-transcribe-arabic-07-2026` | `CohereLabs/cohere-transcribe-03-2026` | [`coherex/__main__.py`](../coherex/__main__.py), [`coherex/asr.py`](../coherex/asr.py) |
| `--language`            | `ar`                                          | *(required, no default)*               | [`coherex/__main__.py`](../coherex/__main__.py)                                         |
| `--vad_method`          | `silero`                                      | `pyannote`                             | [`coherex/__main__.py`](../coherex/__main__.py)                                         |
| `--max_line_width`      | `42`                                          | *(none)*                               | [`coherex/__main__.py`](../coherex/__main__.py)                                         |
| `--max_line_count`      | `2`                                           | *(none)*                               | [`coherex/__main__.py`](../coherex/__main__.py)                                         |
| `--output_dir` / `-o` | `out-ar/`                                     | `.`                                    | [`coherex/__main__.py`](../coherex/__main__.py)                                         |

All other flags are unchanged from upstream (see the main README's
[Common options](../README.md#common-options) table).

With these defaults, the previously-explicit command:

```bash
coherex "{audio_path}" --model CohereLabs/cohere-transcribe-arabic-07-2026 \
  --language ar --vad_method silero --max_line_width 42 --max_line_count 2 \
  -o out-ar/
```

is now equivalent to just:

```bash
coherex "{audio_path}"
```

Every flag above can still be overridden per-run in the usual way, e.g.
`coherex audio.mp3 --language en --model CohereLabs/cohere-transcribe-03-2026 -o out/`
to fall back to the upstream 14-language model for non-Arabic audio.
