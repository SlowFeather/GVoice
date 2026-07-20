from __future__ import annotations

import inspect
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

DEFAULT_SPK_ID = "gvoice_default"


def float_to_pcm16(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32).flatten()
    if samples.size == 0:
        return b""
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


class EngineNotReady(RuntimeError):
    pass


class CosyVoice3Engine:
    """CosyVoice3 流式合成引擎。

    - load() 幂等；start_loading() 在后台线程加载，服务可先监听端口
    - stream() 一次只允许一个合成（GPU 串行），yield pcm_s16le bytes
    - prompt.text 非空走 zero_shot（启动时 add_zero_shot_spk 缓存音色），
      否则走 cross_lingual（无需参考文本）
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._prompt_arg = None
        self._prompt_text_full = ""
        self._is_v3 = False
        self._spk_id = ""
        self._load_lock = threading.Lock()
        self._synth_lock = threading.Lock()
        self._ready = threading.Event()
        self._load_error: BaseException | None = None
        self.sample_rate = 24000

    # ---- 加载 ----

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def state(self) -> str:
        if self._ready.is_set():
            return "READY"
        if self._load_error is not None:
            return "FAILED"
        return "STARTING"

    @property
    def last_error(self) -> str | None:
        return None if self._load_error is None else str(self._load_error)

    def start_loading(self) -> None:
        threading.Thread(target=self._load_in_background, name="cosyvoice3-load", daemon=True).start()

    def _load_in_background(self) -> None:
        try:
            self.load()
        except BaseException as exc:  # noqa: BLE001 - 记录任何加载失败供 wait_ready 抛出
            logger.exception("CosyVoice3 model load failed")
            self._load_error = exc

    def wait_ready(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._ready.is_set():
            if self._load_error is not None:
                raise EngineNotReady(f"model load failed: {self._load_error}") from self._load_error
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise EngineNotReady("model is still loading")
            self._ready.wait(min(0.2, remaining) if remaining is not None else 0.2)
        if self._load_error is not None:
            raise EngineNotReady(f"model load failed: {self._load_error}") from self._load_error

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            self._setup_sys_path()
            model_dir = self._ensure_model()
            self._model = self._create_model(model_dir)
            self.sample_rate = int(getattr(self._model, "sample_rate", 24000))
            self._prepare_prompt()
            if self.cfg.model.warmup:
                self._warmup()
            self._ready.set()
            logger.info(
                "CosyVoice3 engine ready model_dir=%s sample_rate=%d mode=%s load_sec=%.1f",
                model_dir,
                self.sample_rate,
                self._mode_name(),
                time.perf_counter() - started,
            )

    def _setup_sys_path(self) -> None:
        repo = Path(self.cfg.cosyvoice_repo).resolve()
        if not (repo / "cosyvoice").exists():
            raise FileNotFoundError(
                f"CosyVoice repo not found at {repo}; run: "
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor/CosyVoice"
            )
        matcha = repo / "third_party" / "Matcha-TTS"
        if not (matcha / "matcha").exists():
            raise FileNotFoundError(
                f"Matcha-TTS submodule missing at {matcha}; clone the repo with --recursive "
                "or run: git submodule update --init --recursive"
            )
        for entry in (str(matcha), str(repo)):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    def _ensure_model(self) -> Path:
        model_dir = self.cfg.resolve_model_dir()
        if model_dir.exists() and any(model_dir.glob("*.pt")):
            return model_dir
        model_id = self.cfg.resolve_model_id()
        logger.info("Downloading model %s -> %s (source=%s)", model_id, model_dir, self.cfg.model.download_source)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.model.download_source == "huggingface":
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_id, local_dir=str(model_dir))
        else:
            from modelscope import snapshot_download

            snapshot_download(model_id, local_dir=str(model_dir))
        if not any(model_dir.glob("*.pt")):
            raise FileNotFoundError(f"model download incomplete: no *.pt files under {model_dir}")
        return model_dir

    def _create_model(self, model_dir: Path):
        import torch

        fp16 = bool(self.cfg.model.fp16)
        if fp16 and not torch.cuda.is_available():
            logger.warning("CUDA not available; forcing fp16=false (CPU inference will be slow)")
            fp16 = False

        kwargs = {"fp16": fp16}
        if self.cfg.model.load_trt:
            kwargs["load_trt"] = True
        if self.cfg.model.load_vllm:
            kwargs["load_vllm"] = True

        try:
            from cosyvoice.cli.cosyvoice import AutoModel

            model = AutoModel(model_dir=str(model_dir), **kwargs)
        except ImportError:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            logger.warning("AutoModel not available in this CosyVoice checkout; falling back to CosyVoice2")
            model = CosyVoice2(str(model_dir), **kwargs)
        self._is_v3 = type(model).__name__ == "CosyVoice3"
        return model

    def _prepare_prompt(self) -> None:
        wav = self.cfg.prompt.wav.strip()
        if not wav:
            raise ValueError("prompt.wav is required for CosyVoice3 (it defines the output voice)")
        wav_path = Path(wav)
        if not wav_path.exists():
            raise FileNotFoundError(f"prompt wav not found: {wav_path}")

        # 新版 API 直接收文件路径（参数名 prompt_wav），旧版收 16k tensor（prompt_speech_16k）
        sig = inspect.signature(self._model.inference_zero_shot)
        if "prompt_wav" in sig.parameters:
            self._prompt_arg = str(wav_path)
        else:
            from cosyvoice.utils.file_utils import load_wav

            self._prompt_arg = load_wav(str(wav_path), 16000)

        # CosyVoice3 要求 tts_text 或 prompt_text 中包含 <|endofprompt|>（Qwen 特殊 token）。
        # zero_shot 把前缀放进 prompt_text（保留 tts 文本正规化）；cross_lingual 时前缀
        # 由 _inference 拼到正文前（frontend 见到 <|..|> 会跳过正规化，整段直传）
        prompt_text = self.cfg.prompt.text.strip()
        if self._is_v3 and prompt_text:
            self._prompt_text_full = f"{self._v3_system()}<|endofprompt|>{prompt_text}"
        else:
            self._prompt_text_full = prompt_text

        # 缓存音色特征（whisper mel + speech tokenizer + campplus），每句合成免重复提取。
        # prompt 文本为空也可以缓存：cross_lingual 前端会从缓存副本中删掉文本相关字段
        if hasattr(self._model, "add_zero_shot_spk"):
            try:
                ok = self._model.add_zero_shot_spk(self._prompt_text_full, self._prompt_arg, DEFAULT_SPK_ID)
                if ok is not False:
                    self._spk_id = DEFAULT_SPK_ID
                    logger.info("Cached zero-shot speaker id=%s", DEFAULT_SPK_ID)
            except Exception as exc:
                logger.warning("add_zero_shot_spk failed (%s); prompt features will be computed per request", exc)

    def _v3_system(self) -> str:
        return self.cfg.prompt.system.strip() or "You are a helpful assistant."

    def _warmup(self) -> None:
        started = time.perf_counter()
        try:
            for _ in self.stream(self.cfg.model.warmup_text, _skip_ready_check=True):
                pass
            logger.info("Warmup done in %.1fs", time.perf_counter() - started)
        except Exception as exc:
            logger.warning("Warmup synthesis failed: %s", exc)

    def _mode_name(self) -> str:
        if self.cfg.prompt.instruct.strip():
            return "instruct2"
        if self.cfg.prompt.text.strip():
            return "zero_shot"
        return "cross_lingual"

    # ---- 合成 ----

    def stream(
        self,
        text: str,
        speed: float | None = None,
        cancel: threading.Event | None = None,
        *,
        _skip_ready_check: bool = False,
    ) -> Iterator[bytes]:
        if not _skip_ready_check:
            self.wait_ready(timeout=0)
        speed = float(speed) if speed else float(self.cfg.tts.speed)
        with self._synth_lock:
            if cancel is not None and cancel.is_set():
                return
            self._reset_stream_hop()
            gen = self._inference(text, speed)
            try:
                for out in gen:
                    if cancel is not None and cancel.is_set():
                        break
                    pcm = float_to_pcm16(out["tts_speech"].numpy())
                    if pcm:
                        yield pcm
            finally:
                gen.close()
                self._purge_stale_jobs()

    def _purge_stale_jobs(self) -> None:
        # 上游 model.tts 没有 try/finally：生成器被提前关闭（取消/打断）时，llm_job 线程
        # 继续生成 token 且各缓存字典条目永不清理——既泄漏内存又占 GPU 拖慢下一句。
        # 合成结束后清空残留条目：正常完成的请求已被上游 pop，剩下的都是被取消的；
        # pop 掉后僵尸 llm 线程在下一次字典访问时以 KeyError 终止（日志里会有一条
        # 线程异常堆栈，无害）。合成是串行的，此时没有其他在跑的请求。
        model_impl = getattr(self._model, "model", None)
        if model_impl is None:
            return
        names = ("tts_speech_token_dict", "llm_end_dict", "mel_overlap_dict", "hift_cache_dict", "flow_cache_dict")
        dicts = [d for d in (getattr(model_impl, n, None) for n in names) if isinstance(d, dict)]
        stale = set()
        for d in dicts:
            stale.update(d.keys())
        if not stale:
            return
        lock = getattr(model_impl, "lock", None)
        try:
            if lock is not None:
                with lock:
                    for uuid in stale:
                        for d in dicts:
                            d.pop(uuid, None)
            else:
                for uuid in stale:
                    for d in dicts:
                        d.pop(uuid, None)
            logger.info("Purged %d stale synthesis job(s) after cancellation", len(stale))
        except Exception as exc:
            logger.warning("Stale job purge failed: %s", exc)

    def _reset_stream_hop(self) -> None:
        # 上游 CosyVoice2/3 Model 在流式循环里把 self.token_hop_len 翻倍以提升块间质量，
        # 但不跨请求复位，导致后续请求首块越来越大、首包延迟持续劣化。每次合成前复位。
        hop = int(self.cfg.tts.first_chunk_tokens)
        if hop <= 0:
            return
        model_impl = getattr(self._model, "model", None)
        if model_impl is not None and hasattr(model_impl, "token_hop_len"):
            model_impl.token_hop_len = hop

    def _inference(self, text: str, speed: float):
        instruct = self.cfg.prompt.instruct.strip()
        prompt_text = self.cfg.prompt.text.strip()
        common = {"stream": True, "speed": speed, "text_frontend": bool(self.cfg.tts.text_frontend)}
        if instruct:
            # instruct2 不能用缓存音色：缓存会短路 prompt 分支导致 instruct 文本被忽略
            fn = self._model.inference_instruct2
            if self._is_v3:
                instruct = f"{self._v3_system()} {instruct}<|endofprompt|>"
            args = (text, instruct, self._prompt_arg)
        elif prompt_text:
            fn = self._model.inference_zero_shot
            args = (text, self._prompt_text_full, self._prompt_arg)
            if self._spk_id:
                common["zero_shot_spk_id"] = self._spk_id
        else:
            fn = self._model.inference_cross_lingual
            if self._is_v3:
                text = f"{self._v3_system()}<|endofprompt|>{text}"
            args = (text, self._prompt_arg)
            if self._spk_id:
                common["zero_shot_spk_id"] = self._spk_id
        supported = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in common.items() if k in supported}
        return fn(*args, **kwargs)
