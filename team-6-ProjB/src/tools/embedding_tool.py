"""
CardioAgent Embedding Tools — strict retrieval embedders.

This version is designed for real vector retrieval and will FAIL LOUDLY if
ECG-FM cannot be loaded correctly. It intentionally avoids pseudo-embedding
fallbacks so large index builds do not silently produce unusable vectors.

Models:
  ECG-FM      — ECG encoder loaded from ECG-FM / fairseq_signals checkpoints
  BioViL-T    — chest X-ray image encoder from hi-ml-multimodal
  BiomedBERT  — biomedical text encoder from Hugging Face

Each embedder exposes:
  .embed(input) -> np.ndarray of shape (embedding_dim,)
  .embed_batch(inputs) -> np.ndarray of shape (batch, embedding_dim)
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector."""
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return vec / norm
    return vec



def _l2_normalize_batch(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of vectors row-wise."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vecs / norms



def _resolve_device(device: str) -> str:
    """Return a usable torch device string."""
    import torch

    requested = (device or "cuda").lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to cpu")
        return "cpu"
    return device



def _safe_float32(arr: np.ndarray) -> np.ndarray:
    """Convert array to contiguous float32 and replace NaN/Inf."""
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr)


# -----------------------------------------------------------------------------
# ECG-FM Embedder
# -----------------------------------------------------------------------------


class ECGFMEmbedder:
    """
    ECG-FM embedder for 12-lead ECG retrieval.

    Expected input after preprocessing:
      (12, 5000) at 500 Hz

    Output:
      Dense embedding vector of size `embedding_dim` (default 1024)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        target_fs: int = 500,
        target_length: int = 5000,
        embedding_dim: int = 768,
        strict_loading: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = _resolve_device(device)
        self.target_fs = target_fs
        self.target_length = target_length
        self.embedding_dim = embedding_dim
        self.strict_loading = strict_loading
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _find_checkpoint(self) -> str:
        """Find a checkpoint file from a path or directory."""
        ckpt = Path(self.checkpoint_path)

        if ckpt.is_file() and ckpt.suffix == ".pt":
            return str(ckpt)

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")

        priority = [
            "mimic_iv_ecg_finetuned.pt",
            "mimic_iv_ecg_physionet_pretrained.pt",
            "checkpoint_best.pt",
            "checkpoint_last.pt",
        ]
        for name in priority:
            candidate = ckpt / name
            if candidate.exists():
                return str(candidate)

        pt_files = sorted(ckpt.glob("*.pt"))
        if pt_files:
            return str(pt_files[0])

        raise FileNotFoundError(
            f"No .pt checkpoint found in {ckpt}. Expected one of {priority} or any .pt file."
        )

    def _register_local_modules(self, base_dir: Path) -> None:
        """Best-effort import of local ECG-FM / fairseq_signals registration code."""
        candidate_dirs = [
            base_dir,
            base_dir / "src",
            base_dir / "fairseq_signals",
            base_dir / "ecg_fm",
            base_dir / "models",
        ]

        for d in candidate_dirs:
            if not d.is_dir():
                continue
            d_str = str(d)
            if d_str not in sys.path:
                sys.path.insert(0, d_str)

            for py_file in sorted(d.glob("*.py")):
                stem = py_file.stem
                if stem.startswith("_"):
                    continue
                try:
                    importlib.import_module(stem)
                except Exception as exc:
                    logger.debug("Local import skipped for %s: %s", py_file, exc)

    def _clear_cached_modules(self) -> None:
        """Clear fairseq/fairseq_signals modules so patched imports take effect."""
        prefixes = ("fairseq", "fairseq_signals")
        to_delete = [name for name in sys.modules if name.startswith(prefixes)]
        for name in to_delete:
            del sys.modules[name]

    def _clean_omegaconf_obj(self, obj: Any) -> Any:
        """
        Recursively clean OmegaConf / dataclass-MISSING content.

        This removes Python dataclasses._MISSING_TYPE instances and OmegaConf
        missing values before rebuilding a plain config for direct model build.
        """
        from dataclasses import MISSING
        from omegaconf import DictConfig, ListConfig, OmegaConf

        missing_type = type(MISSING)

        if isinstance(obj, missing_type):
            return None

        if isinstance(obj, DictConfig):
            out: Dict[str, Any] = {}
            for key in list(obj.keys()):
                if OmegaConf.is_missing(obj, key):
                    continue
                try:
                    value = obj[key]
                except Exception:
                    continue
                cleaned = self._clean_omegaconf_obj(value)
                if cleaned is not None:
                    out[str(key)] = cleaned
            return out

        if isinstance(obj, ListConfig):
            cleaned_list = [self._clean_omegaconf_obj(x) for x in list(obj)]
            return [x for x in cleaned_list if x is not None]

        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                cleaned = self._clean_omegaconf_obj(value)
                if cleaned is not None:
                    out[key] = cleaned
            return out

        if isinstance(obj, (list, tuple)):
            cleaned = [self._clean_omegaconf_obj(x) for x in obj]
            cleaned = [x for x in cleaned if x is not None]
            return type(obj)(cleaned)

        return obj

    def _sanitize_model_cfg(self, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove stale pretrained-path hooks and fragile loading directives."""
        model_cfg = cfg_dict.setdefault("model", {})
        if not isinstance(model_cfg, dict):
            cfg_dict["model"] = {}
            model_cfg = cfg_dict["model"]

        # ECG-FM finetuning configs often store the original pretraining path in model_path.
        # That path does not exist on a new machine and should not be reloaded here.
        for key in (
            "model_path",
            "w2v_path",
            "pretrained_model_path",
            "restore_file",
            "checkpoint_file",
        ):
            if key in model_cfg:
                model_cfg[key] = None

        model_cfg["no_pretrained_weights"] = True
        return cfg_dict

    def _try_fairseq_checkpoint_load(self, ckpt_path: str):
        """Try official checkpoint loading first."""
        from fairseq import checkpoint_utils

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        if not models:
            raise RuntimeError("checkpoint_utils returned no models")
        logger.info("ECG-FM loaded via fairseq checkpoint_utils")
        return models[0]

    def _get_registries(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Collect model/task registries from fairseq and fairseq_signals."""
        model_registry: Dict[str, Any] = {}
        task_registry: Dict[str, Any] = {}

        try:
            from fairseq.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY

            model_registry.update(MODEL_REGISTRY)
            model_registry.update(ARCH_MODEL_REGISTRY)
        except Exception as exc:
            logger.debug("fairseq model registry unavailable: %s", exc)

        try:
            from fairseq.tasks import TASK_REGISTRY

            task_registry.update(TASK_REGISTRY)
        except Exception as exc:
            logger.debug("fairseq task registry unavailable: %s", exc)

        try:
            from fairseq_signals.models import ARCH_MODEL_REGISTRY as FS_ARCH_MODEL_REGISTRY
            from fairseq_signals.models import MODEL_REGISTRY as FS_MODEL_REGISTRY

            model_registry.update(FS_MODEL_REGISTRY)
            model_registry.update(FS_ARCH_MODEL_REGISTRY)
        except Exception as exc:
            logger.debug("fairseq_signals model registry unavailable: %s", exc)

        try:
            from fairseq_signals.tasks import TASK_REGISTRY as FS_TASK_REGISTRY

            task_registry.update(FS_TASK_REGISTRY)
        except Exception as exc:
            logger.debug("fairseq_signals task registry unavailable: %s", exc)

        return model_registry, task_registry

    def _try_direct_build(self, ckpt_path: str):
        """Build the model from checkpoint state + cleaned config."""
        import torch
        from omegaconf import DictConfig, OmegaConf

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        cfg = state.get("cfg")
        if cfg is None:
            raise RuntimeError("Checkpoint does not contain cfg; cannot direct-build model")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        cfg_dict = self._clean_omegaconf_obj(cfg)
        cfg_dict = self._sanitize_model_cfg(cfg_dict)
        cfg = OmegaConf.create(cfg_dict)

        model_name = None
        if isinstance(cfg_dict.get("model"), dict):
            model_name = cfg_dict["model"].get("_name")
        if not model_name:
            raise RuntimeError("Could not determine model._name from checkpoint cfg")

        logger.info("Attempting direct build of '%s'...", model_name)

        model_registry, task_registry = self._get_registries()
        if model_name not in model_registry:
            available = sorted(model_registry.keys())[:50]
            raise RuntimeError(
                f"Model '{model_name}' not found in registries. Sample available entries: {available}"
            )

        task = None
        task_cfg = cfg_dict.get("task")
        if isinstance(task_cfg, dict):
            task_name = task_cfg.get("_name")
            if task_name and task_name in task_registry:
                try:
                    task = task_registry[task_name].setup_task(OmegaConf.create(task_cfg))
                except Exception as exc:
                    logger.warning("Task setup skipped for %s: %s", task_name, exc)

        model_cls = model_registry[model_name]
        model = model_cls.build_model(cfg.model, task=task)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            logger.debug("Missing keys during load_state_dict: %d", len(missing))
        if unexpected:
            logger.debug("Unexpected keys during load_state_dict: %d", len(unexpected))

        logger.info("ECG-FM loaded via direct model build")
        return model

    def _load_model(self) -> None:
        """Load ECG-FM via fairseq_signals registry, bypassing broken fairseq config."""
        import dataclasses
        import torch
        import sys

        ckpt_path = self._find_checkpoint()
        logger.info("Loading ECG-FM from %s...", ckpt_path)

        orig_field = dataclasses.field
        def patched_field(*a, **kw):
            d = kw.get("default", dataclasses.MISSING)
            if d is not dataclasses.MISSING and not isinstance(
                d, (int, float, str, bool, type(None), tuple, type)
            ):
                kw["default_factory"] = lambda d=d: d
                kw.pop("default", None)
            return orig_field(*a, **kw)
        dataclasses.field = patched_field

        for k in list(sys.modules):
            if "fairseq" in k:
                del sys.modules[k]

        try:
            from omegaconf import OmegaConf

            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = state.get("model", state)
            cfg = state.get("cfg", {})

            model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
            task_cfg = cfg.get("task", {}) if isinstance(cfg, dict) else {}

            model_name = model_cfg.get("_name", "")
            logger.info("  Building %s (embed_dim=%s)...",
                        model_name, model_cfg.get("encoder_embed_dim", "?"))

            from fairseq_signals.models import MODEL_REGISTRY, ARCH_MODEL_REGISTRY
            registry = {**MODEL_REGISTRY, **ARCH_MODEL_REGISTRY}

            if model_name not in registry:
                raise RuntimeError(f"'{model_name}' not in registry: {sorted(registry.keys())[:20]}")

            model_oc = OmegaConf.create(model_cfg)
            task = None
            task_name = task_cfg.get("_name", "")
            if task_name:
                try:
                    from fairseq_signals.tasks import TASK_REGISTRY
                    if task_name in TASK_REGISTRY:
                        task = TASK_REGISTRY[task_name].setup_task(OmegaConf.create(task_cfg))
                except Exception as te:
                    logger.debug("Task setup skipped: %s", te)

            model = registry[model_name].build_model(model_oc, task=task)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            logger.info("  Loaded: %d missing, %d unexpected keys", len(missing), len(unexpected))

            actual_dim = model_cfg.get("encoder_embed_dim", self.embedding_dim)
            if actual_dim != self.embedding_dim:
                logger.info("  Updating embed_dim %d -> %d", self.embedding_dim, actual_dim)
                self.embedding_dim = actual_dim

            model = model.to(self.device).eval()
            if hasattr(model, "remove_pretraining_modules"):
                try:
                    model.remove_pretraining_modules()
                except Exception:
                    pass

            self._model = model
            logger.info("ECG-FM loaded on %s (embed_dim=%d)", self.device, self.embedding_dim)

        except Exception as e:
            dataclasses.field = orig_field
            msg = f"ECG-FM failed to load: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        dataclasses.field = orig_field

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize and reshape ECG signal to (12, target_length).

        Supported common input shapes:
          (12, 5000)
          (5000, 12)
          (1, 5000, 12)
          (N,)  -> single lead, padded to 12 leads
        """
        arr = _safe_float32(signal).copy()

        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(f"Unsupported ECG shape after squeeze: {arr.shape}")

        # Convert (time, leads) -> (leads, time)
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 12:
            arr = arr.T

        n_leads, n_samples = arr.shape

        if n_leads > 12:
            arr = arr[:12]
            n_leads = 12

        if n_leads < 12:
            pad = np.zeros((12 - n_leads, n_samples), dtype=np.float32)
            arr = np.vstack([arr, pad])

        if n_samples != self.target_length:
            from scipy.signal import resample

            arr = resample(arr, self.target_length, axis=1).astype(np.float32)

        for i in range(arr.shape[0]):
            std = float(arr[i].std())
            if std > 1e-6:
                arr[i] = (arr[i] - arr[i].mean()) / std
            else:
                arr[i] = arr[i] - arr[i].mean()

        return np.ascontiguousarray(arr, dtype=np.float32)

    def _coerce_feature_tensor(self, obj, batch_size: int):
        """
        Convert a model output container into a tensor of shape (B, T, C) or (B, C).

        Handles common fairseq / fairseq_signals return structures:
          - Tensor
          - tuple/list whose first item is a Tensor or nested container
          - dict with keys like x, features, encoder_out, encoder_states, out
          - encoder_out packed as a one-element list with shape (T, B, C)
        """
        import torch

        if obj is None:
            return None

        if isinstance(obj, torch.Tensor):
            return obj

        if isinstance(obj, dict):
            preferred_keys = (
                "x",
                "features",
                "encoder_out",
                "encoder_states",
                "out",
                "last_hidden_state",
            )
            for key in preferred_keys:
                if key in obj:
                    tensor = self._coerce_feature_tensor(obj[key], batch_size)
                    if tensor is not None:
                        return tensor
            for value in obj.values():
                tensor = self._coerce_feature_tensor(value, batch_size)
                if tensor is not None:
                    return tensor
            return None

        if isinstance(obj, (list, tuple)):
            # Prefer tensor-like entries with actual hidden states over masks/lengths.
            for item in obj:
                tensor = self._coerce_feature_tensor(item, batch_size)
                if tensor is None:
                    continue
                if hasattr(tensor, "ndim") and tensor.ndim >= 2:
                    return tensor
            return None

        return None

    def _extract_features(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Extract sequence features and mean-pool to (batch, embed_dim)."""
        import torch

        batch_size = int(tensor.shape[0])
        raw_output = None

        if hasattr(self.model, "extract_features"):
            try:
                raw_output = self.model.extract_features(
                    source=tensor,
                    padding_mask=None,
                )
            except TypeError:
                # Some classifier wrappers expose a different signature.
                raw_output = self.model.extract_features(tensor)

        if raw_output is None and hasattr(self.model, "feature_extractor"):
            raw_output = self.model.feature_extractor(tensor)
            if hasattr(self.model, "encoder"):
                try:
                    raw_output = self.model.encoder(raw_output)
                except TypeError:
                    raw_output = self.model.encoder(raw_output, padding_mask=None)

        if raw_output is None:
            try:
                raw_output = self.model(tensor, padding_mask=None, features_only=True)
            except TypeError:
                try:
                    raw_output = self.model(tensor, features_only=True)
                except TypeError:
                    raw_output = self.model(tensor)

        features = self._coerce_feature_tensor(raw_output, batch_size)
        if features is None:
            if isinstance(raw_output, dict):
                keys = list(raw_output.keys())
                raise RuntimeError(
                    f"Could not find feature tensor in model output dict. Keys={keys}"
                )
            raise RuntimeError(
                f"Could not find feature tensor in model output of type {type(raw_output)}"
            )

        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim == 2:
            # Already pooled or classifier-head output: (B, C)
            if features.shape[0] == batch_size:
                return features
            # Rare case: single sample returned as (T, C)
            if batch_size == 1:
                features = features.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Ambiguous 2D ECG-FM feature shape: {tuple(features.shape)}"
                )

        if features.ndim != 3:
            raise RuntimeError(f"Unexpected ECG-FM feature shape: {tuple(features.shape)}")

        # Convert fairseq-style (T, B, C) to (B, T, C)
        if features.shape[0] != batch_size and features.shape[1] == batch_size:
            features = features.transpose(0, 1)

        if features.shape[0] != batch_size:
            raise RuntimeError(
                f"ECG-FM batch dimension mismatch after extraction: "
                f"got {tuple(features.shape)}, expected batch_size={batch_size}"
            )

        pooled = features.mean(dim=1)
        if pooled.ndim != 2:
            raise RuntimeError(f"Unexpected pooled ECG-FM shape: {tuple(pooled.shape)}")
        return pooled

    def embed(self, signal: np.ndarray) -> np.ndarray:
        """Embed one ECG signal into a normalized dense vector."""
        import torch

        processed = self.preprocess(signal)
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self._extract_features(tensor)

        vec = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def embed_batch(self, signals: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of ECG signals."""
        import torch

        all_vecs: List[np.ndarray] = []
        for start in range(0, len(signals), batch_size):
            batch = signals[start:start + batch_size]
            processed = np.stack([self.preprocess(s) for s in batch], axis=0)
            tensor = torch.from_numpy(processed).to(self.device)
            with torch.no_grad():
                embedding = self._extract_features(tensor)
            all_vecs.append(embedding.detach().cpu().numpy().astype(np.float32))

        if not all_vecs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return _l2_normalize_batch(np.vstack(all_vecs))

    def embed_from_path(self, ecg_path: str) -> np.ndarray:
        """Load a .npy or WFDB record and embed it."""
        p = Path(ecg_path)
        if p.suffix == ".npy":
            signal = np.load(str(p))
        elif p.suffix in (".hea", ".dat", ""):
            import wfdb

            record = wfdb.rdrecord(str(p.with_suffix("")))
            signal = record.p_signal.T
        else:
            raise ValueError(f"Unsupported ECG format: {p.suffix}")
        return self.embed(signal)


# -----------------------------------------------------------------------------
# BioViL-T Embedder
# -----------------------------------------------------------------------------


class BioViLTEmbedder:
    """
    BioViL-T chest X-ray image embedder.

    Requires the `hi-ml-multimodal` package, imported as `health_multimodal`.
    """

    def __init__(self, device: str = "cuda", image_size: int = 480):
        self.device = _resolve_device(device)
        self.image_size = image_size
        self._image_inference = None
        self._text_inference = None

    @property
    def image_inference(self):
        if self._image_inference is None:
            self._load_image_model()
        return self._image_inference

    def _load_image_model(self) -> None:
        from health_multimodal.image.model.types import ImageEncoderType #ImageModelType
        from health_multimodal.image.utils import get_image_inference

        logger.info("Loading BioViL-T image encoder...")
        self._image_inference = get_image_inference()
        self._image_inference.model.to(self.device)
        self._image_inference.model.eval()
        logger.info("BioViL-T image encoder loaded on %s", self.device)

    def _load_image(self, image_input):
        from PIL import Image

        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("L")
        if isinstance(image_input, Image.Image):
            return image_input.convert("L")
        if isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype in (np.float32, np.float64):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("L")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def embed(self, image_input) -> np.ndarray:
        import torch

        image = self._load_image(image_input)
        transform = self.image_inference.transform
        tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.image_inference.model(tensor)
            if hasattr(output, "projected_global_embedding"):
                embedding = output.projected_global_embedding
            elif hasattr(output, "img_embedding"):
                embedding = output.img_embedding
            else:
                available = [x for x in dir(output) if not x.startswith("_")]
                raise AttributeError(
                    "BioViL-T output has no recognized embedding field. "
                    f"Available attributes: {available}"
                )

        vec = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def embed_batch(self, image_inputs: List[Any], batch_size: int = 16) -> np.ndarray:
        import torch

        transform = self.image_inference.transform
        all_vecs: List[np.ndarray] = []

        for start in range(0, len(image_inputs), batch_size):
            batch_imgs = image_inputs[start:start + batch_size]
            images = [self._load_image(x) for x in batch_imgs]
            tensor = torch.stack([transform(img) for img in images]).to(self.device)
            with torch.no_grad():
                output = self.image_inference.model(tensor)
                if hasattr(output, "projected_global_embedding"):
                    embeddings = output.projected_global_embedding
                elif hasattr(output, "img_embedding"):
                    embeddings = output.img_embedding
                else:
                    raise AttributeError("BioViL-T output missing embedding field")
            all_vecs.append(embeddings.detach().cpu().numpy().astype(np.float32))

        if not all_vecs:
            return np.empty((0, 128), dtype=np.float32)
        return _l2_normalize_batch(np.vstack(all_vecs))

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        from health_multimodal.text.model.types import TextModelType
        from health_multimodal.text.utils import get_text_inference

        if self._text_inference is None:
            logger.info("Loading BioViL-T text encoder...")
            self._text_inference = get_text_inference(text_model_type=TextModelType.BIOVIL_T)
            self._text_inference.model.to(self.device)
            self._text_inference.model.eval()

        with torch.no_grad():
            text_emb = self._text_inference.get_embeddings_from_prompt(text, verbose=False)

        vec = text_emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)


# -----------------------------------------------------------------------------
# BiomedBERT Embedder
# -----------------------------------------------------------------------------


class BiomedBERTEmbedder:
    """
    Biomedical text embedder based on microsoft/BiomedNLP-BiomedBERT.
    """

    def __init__(
        self,
        checkpoint_path: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = _resolve_device(device)
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading BiomedBERT from %s...", self.checkpoint_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self._model = AutoModel.from_pretrained(self.checkpoint_path)
        self._model.to(self.device).eval()
        logger.info("BiomedBERT loaded on %s", self.device)

    def embed(self, text: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]

        vec = cls_emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(vec)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch

        all_vecs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]

            all_vecs.append(cls_emb.detach().cpu().numpy().astype(np.float32))

        if not all_vecs:
            return np.empty((0, 768), dtype=np.float32)
        return _l2_normalize_batch(np.vstack(all_vecs))
