"""
Global model singleton for Leva-TTS.

Python's import system guarantees this module is loaded exactly ONCE per
process, regardless of how app.py is imported (__main__ vs app vs any other
name). Storing the model here ensures a single shared instance.
"""
import os
from pathlib import Path

# ── Compatibility patch: transformers 4.41 calls torch.isin with keyword args
#    'elements='/'test_elements=' that torch 2.3 does not accept. This breaks
#    XTTS streaming (inference_stream). Translate kwargs → positional.
def _patch_torch_isin():
    import torch
    if getattr(torch.isin, '_leva_patched', False):
        return
    _orig = torch.isin
    def _isin(*args, **kwargs):
        if 'elements' in kwargs or 'test_elements' in kwargs:
            el = kwargs.pop('elements', args[0] if len(args) > 0 else None)
            te = kwargs.pop('test_elements', args[1] if len(args) > 1 else None)
            # torch.isin needs at least one tensor — wrap scalars if needed
            if not torch.is_tensor(el) and not torch.is_tensor(te):
                el = torch.tensor(el)
            return _orig(el, te, **kwargs)
        # also handle positional scalar/scalar
        if len(args) == 2 and not torch.is_tensor(args[0]) and not torch.is_tensor(args[1]):
            return _orig(torch.tensor(args[0]), args[1], **kwargs)
        return _orig(*args, **kwargs)
    _isin._leva_patched = True
    torch.isin = _isin

_patch_torch_isin()


_MODEL    = None
_CONFIG   = None
_COND     = {}   # {wav_path: (gpt_cond, spk_emb)}
_READY    = False
_ERR      = None


def _find_ckpt(d: str):
    bests = sorted(Path(d).rglob("best_model.pth"),
                   key=lambda f: f.stat().st_mtime)
    if bests: return str(bests[-1])
    pths  = sorted(Path(d).rglob("*.pth"),
                   key=lambda f: f.stat().st_mtime)
    return str(pths[-1]) if pths else None


def load(checkpoint_dir: str = "checkpoints", device: str = "cuda"):
    """
    Load XTTS-v2 + fine-tuned checkpoint into module globals.
    Safe to call multiple times — returns immediately if already loaded.
    """
    global _MODEL, _CONFIG, _READY, _ERR
    if _READY:
        return _MODEL, _CONFIG
    if _ERR:
        raise RuntimeError(f"Model loading previously failed: {_ERR}")

    import torch

    cache = Path(os.environ.get(
        "COQUI_MODEL_PATH", Path.home() / ".local/share/tts"))
    xtts  = cache / "tts_models--multilingual--multi-dataset--xtts_v2"

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        cfg = XttsConfig()
        cfg.load_json(str(xtts / "config.json"))

        mdl = Xtts.init_from_config(cfg)
        mdl.load_checkpoint(cfg, checkpoint_dir=str(xtts), eval=True)

        ckpt = _find_ckpt(checkpoint_dir)
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("model", state)
            cleaned = {
                (k[len("xtts.gpt."):] if k.startswith("xtts.gpt.")
                 else k[len("gpt."):] if k.startswith("gpt.") else k): v
                for k, v in state.items()
            }
            mdl.gpt.load_state_dict(cleaned, strict=False)

        # IMPORTANT: do NOT chain .to().eval() — XTTS .eval() returns None,
        # which would silently set the model to None.
        mdl.to(device)
        mdl.eval()
        assert mdl is not None, 'model became None'
        _MODEL, _CONFIG = mdl, cfg
        _READY = True
        return _MODEL, _CONFIG

    except Exception as exc:
        _ERR = str(exc)
        raise RuntimeError(f"Model loading failed: {exc}") from exc


def get():
    """Return (model, config). Raises if not loaded."""
    if not _READY:
        raise RuntimeError(
            "Model not loaded. Call leva_tts._model.load() first.")
    return _MODEL, _CONFIG


def get_conditioning(ref_wav: str):
    """Return cached (gpt_cond, spk_emb) for a reference WAV."""
    if ref_wav not in _COND:
        mdl, _ = get()
        gpt, emb = mdl.get_conditioning_latents(audio_path=[ref_wav])
        _COND[ref_wav] = (gpt, emb)
    return _COND[ref_wav]
