"""
XTTS-v2 fine-tuning script for Saudi Arabic / English code-switching TTS.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from omegaconf import OmegaConf
import wandb

# ── Patched XttsAudioConfig ───────────────────────────────────────────────────
# GPTTrainer.__init__ (line 195) accesses config.audio.dvae_sample_rate which
# is missing from XttsAudioConfig in TTS==0.22.0. Subclass it to add the field.
from TTS.tts.models.xtts import XttsAudioConfig as _XttsAudioConfigBase


@dataclass
class XttsAudioConfig(_XttsAudioConfigBase):
    """XttsAudioConfig patched for TTS==0.22.0 GPTTrainer compatibility."""
    dvae_sample_rate: int = 22050   # needed by GPTTrainer line 195


logger = logging.getLogger(__name__)

# ── Custom TTS formatter for leva-tts metadata files ─────────────────────────
def _leva_tts_formatter(root_path, meta_file, ignored_speakers=None):
    """
    Flexible formatter that handles both our metadata layouts:
      A) wavs/spk/file.wav|text      (2 cols, full relative path — synthetic data)
      B) stem|text|text              (3 cols, ID only — preprocessed data, ljspeech-style)
    """
    import os
    txt_path = os.path.join(root_path, meta_file)
    items = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|")
            if len(cols) < 2:
                continue
            col0 = cols[0].strip()
            text = cols[-1].strip()   # last column is always the transcript
            if not text:
                continue
            # Determine WAV path
            if ".wav" in col0 or "/" in col0:
                # Layout A: col0 is already a relative path
                wav_file = os.path.join(root_path, col0)
            else:
                # Layout B: col0 is a stem ID (ljspeech-style)
                wav_file = os.path.join(root_path, "wavs", col0 + ".wav")
            if not os.path.isfile(wav_file):
                continue
            items.append({
                "text":         text,
                "audio_file":   wav_file,
                "speaker_name": "leva_tts",
                "root_path":    root_path,
                "language":     "ar",
            })
    return items



# ─────────────────────────────────────────────────────────────────────────────

def _resolve_xtts_dir() -> Path:
    """Return the local XTTS-v2 cache directory, downloading if needed."""
    from TTS.utils.manage import ModelManager
    cache_dir = Path(os.environ.get("COQUI_MODEL_PATH",
                                    Path.home() / ".local/share/tts"))
    xtts_dir  = cache_dir / "tts_models--multilingual--multi-dataset--xtts_v2"

    if not xtts_dir.exists():
        logger.info("Downloading XTTS-v2 pre-trained weights …")
        try:
            ModelManager().download_model(
                "tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception as e:
            logger.warning(f"ModelManager failed ({e}). Trying HF hub …")
            from huggingface_hub import snapshot_download
            xtts_dir = Path(
                snapshot_download("coqui/XTTS-v2", local_dir=str(xtts_dir)))
    return xtts_dir


def _ensure_asset(xtts_dir: Path, filename: str) -> Path:
    """Download a missing XTTS-v2 asset from HuggingFace (Coqui CDN is dead)."""
    local = xtts_dir / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading {filename} from coqui/XTTS-v2 …")
    try:
        path = hf_hub_download(repo_id="coqui/XTTS-v2", filename=filename,
                               local_dir=str(xtts_dir))
        logger.info(f"  → {path}")
        return Path(path)
    except Exception as e:
        logger.warning(f"{filename} download failed: {e}")
        return local


def build_dataset_configs(datasets_cfg: List[Dict]) -> List:
    from TTS.config.shared_configs import BaseDatasetConfig
    return [
        BaseDatasetConfig(
            formatter      = d.get("formatter", "ljspeech"),
            dataset_name   = d["name"],
            path           = d["path"],
            meta_file_train= d.get("meta_file", "metadata.csv"),
            language       = d.get("language", "ar"),
        )
        for d in datasets_cfg
    ]


# ─────────────────────────────────────────────────────────────────────────────

def run_finetuning(cfg_path: str):
    """Main fine-tuning entry point."""
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs, GPTTrainer, GPTTrainerConfig)
    from trainer import Trainer, TrainerArgs

    # ── Load YAML config ──────────────────────────────────────────────────────
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)

    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Resolve XTTS-v2 local checkpoint directory ────────────────────────────
    xtts_dir      = _resolve_xtts_dir()
    tokenizer_file = str(xtts_dir / "vocab.json")
    model_pth      = xtts_dir / "model.pth"
    mel_norm_path  = _ensure_asset(xtts_dir, "mel_stats.pth")
    dvae_path      = _ensure_asset(xtts_dir, "dvae.pth")

    # ── W&B init ─────────────────────────────────────────────────────────────
    BATCH_SIZE  = cfg.get("batch_size",       4)
    GRAD_ACCUM  = cfg.get("grad_accum_steps", 8)
    EPOCHS      = cfg.get("epochs",           30)
    wandb.init(
        project = cfg.get("project_name", "saudi-tts"),
        name    = cfg.get("run_name",     "saudi_xtts_ft"),
        config  = {
            "epochs":              EPOCHS,
            "batch_size":          BATCH_SIZE,
            "grad_accum_steps":    GRAD_ACCUM,
            "effective_batch":     BATCH_SIZE * GRAD_ACCUM,
            "lr":                  float(cfg.get("lr", 5e-6)),
            "optimizer":           cfg.get("optimizer", "AdamW"),
            "lr_scheduler":        cfg.get("lr_scheduler", "MultiStepLR"),
            "sample_rate":         cfg.get("audio", {}).get("sample_rate", 22050),
            "datasets":            [d["name"] for d in cfg.get("datasets", [])],
        },
        resume  = "allow",
    )
    logger.info(f"W&B run: {wandb.run.url}")

    # ── GPT model args ────────────────────────────────────────────────────────
    # Must use GPTArgs (not bare XttsArgs) — GPTTrainer checks xtts_checkpoint,
    # dvae_checkpoint, mel_norm_file at __init__ time.
    # Token counts verified from the downloaded config.json:
    #   gpt_num_audio_tokens=1026, start=1024, stop=1025
    ma = cfg.get("model_args", {})
    model_args = GPTArgs(
        tokenizer_file             = tokenizer_file,
        xtts_checkpoint            = str(model_pth),
        mel_norm_file              = str(mel_norm_path),
        dvae_checkpoint            = str(dvae_path),
        gpt_num_audio_tokens       = ma.get("gpt_num_audio_tokens",       1026),
        gpt_start_audio_token      = ma.get("gpt_start_audio_token",      1024),
        gpt_stop_audio_token       = ma.get("gpt_stop_audio_token",       1025),
        gpt_use_masking_gt_prompt_approach = ma.get(
            "gpt_use_masking_gt_prompt_approach", True),
        gpt_use_perceiver_resampler= ma.get("gpt_use_perceiver_resampler", True),
        max_conditioning_length    = ma.get("max_conditioning_length",    132300),
    )

    # ── Audio config (patched subclass adds dvae_sample_rate) ────────────────
    ac   = cfg.get("audio", {})
    sr   = ac.get("sample_rate", 22050)
    audio_config = XttsAudioConfig(
        sample_rate        = sr,
        output_sample_rate = ac.get("output_sample_rate", 24000),
        dvae_sample_rate   = sr,   # required by GPTTrainer line 195
    )

    # ── Dataset configs ───────────────────────────────────────────────────────
    dataset_cfgs = build_dataset_configs(cfg.get("datasets", []))

    # ── Speaker reference ─────────────────────────────────────────────────────
    speaker_ref = cfg.get("speaker_reference", None)
    if speaker_ref and not Path(speaker_ref).exists():
        logger.warning(f"Speaker reference not found: {speaker_ref}")
        speaker_ref = None

    # ── Test sentences for TensorBoard audio logging ──────────────────────────
    test_sents = [
        {
            "text":        s["text"],
            "speaker_wav": [speaker_ref] if speaker_ref else [],
            "language":    s.get("language", "ar"),
        }
        for s in cfg.get("test_sentences", [])
    ]

    # ── GPTTrainerConfig ──────────────────────────────────────────────────────
    opt         = cfg.get("optimizer_params", {})
    sch         = cfg.get("lr_scheduler_params", {})

    gpt_config = GPTTrainerConfig(
        epochs              = EPOCHS,
        output_path         = str(output_path),
        model_args          = model_args,
        run_name            = cfg.get("run_name",    "saudi_xtts_ft"),
        project_name        = cfg.get("project_name", "saudi-tts"),
        run_description     = "Saudi Arabic / English code-switching XTTS-v2",
        dashboard_logger    = cfg.get("dashboard_logger", "wandb"),
        audio               = audio_config,
        batch_size          = BATCH_SIZE,
        batch_group_size    = 48,
        eval_batch_size     = cfg.get("eval_batch_size",    4),
        num_loader_workers  = cfg.get("num_loader_workers", 8),
        eval_split_max_size = cfg.get("eval_split_max_size", 128),
        eval_split_size     = cfg.get("eval_split_size",    0.01),
        print_step          = cfg.get("print_step",   50),
        plot_step           = cfg.get("plot_step",   200),
        log_model_step      = cfg.get("log_model_step", 1000),
        save_step           = cfg.get("save_step",   2000),
        save_n_checkpoints  = cfg.get("save_n_checkpoints", 3),
        save_checkpoints    = cfg.get("save_checkpoints", False),
        target_loss         = "loss",
        print_eval          = False,
        optimizer           = cfg.get("optimizer", "AdamW"),
        optimizer_wd_only_on_weights = cfg.get("optimizer_wd_only_on_weights", True),
        optimizer_params    = {
            "betas":        list(opt.get("betas", [0.9, 0.96])),
            "eps":          float(opt.get("eps", 1e-8)),
            "weight_decay": float(opt.get("weight_decay", 1e-2)),
        },
        lr              = float(cfg.get("lr", 5e-6)),
        lr_scheduler    = cfg.get("lr_scheduler", "MultiStepLR"),
        lr_scheduler_params = {
            "milestones": list(sch.get("milestones", [50000, 150000, 300000])),
            "gamma":      float(sch.get("gamma", 0.5)),
            "last_epoch": int(sch.get("last_epoch", -1)),
        },
        test_sentences  = test_sents,
        datasets        = dataset_cfgs,
    )

    # ── Init model ────────────────────────────────────────────────────────────
    logger.info("Initialising GPTTrainer from pre-trained XTTS-v2 …")
    model = GPTTrainer.init_from_config(gpt_config)
    logger.info("Model initialised.")

    # ── Load training samples ─────────────────────────────────────────────────
    from TTS.tts.datasets.formatters import register_formatter
    register_formatter("leva_tts", _leva_tts_formatter)

    train_samples, eval_samples = load_tts_samples(
        gpt_config.datasets,
        eval_split          = True,
        eval_split_max_size = gpt_config.eval_split_max_size,
        eval_split_size     = gpt_config.eval_split_size,
    )
    logger.info(f"Training: {len(train_samples):,}  Eval: {len(eval_samples):,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        TrainerArgs(
            restore_path    = None,
            skip_train_epoch= False,
            start_with_eval = (GRAD_ACCUM == 0),
            grad_accum_steps= GRAD_ACCUM,
        ),
        gpt_config,
        output_path  = str(output_path),
        model        = model,
        train_samples= train_samples,
        eval_samples = eval_samples,
        # callbacks omitted — uses Trainer default (empty); TensorBoard via dashboard_logger
    )

    # Patch checkpoint saves to avoid running out of disk space.
    # Each checkpoint is ~5.3 GB; the default behaviour writes new then deletes
    # old, requiring 2× free space. We pre-delete before every write instead.
    # best_model.pth is created as a symlink (not a 5.3 GB copy).
    import trainer.trainer as _trainer_mod
    import trainer.io as _trainer_io
    from trainer.io import sort_checkpoints
    import fsspec, os

    def _pre_delete_pths(output_folder):
        """Delete all .pth files in the run directory before writing a new one."""
        fs = fsspec.get_mapper(str(output_folder)).fs
        for f in fs.glob(os.path.join(str(output_folder), "*.pth")):
            fs.rm(f)
            logger.info(f"Pre-deleted to free disk space: {f}")

    _orig_save = _trainer_mod.save_checkpoint
    def _disk_safe_save(config, model, output_folder, *, current_step, epoch,
                        save_n_checkpoints=None, **kwargs):
        _pre_delete_pths(output_folder)
        _orig_save(config, model, output_folder, current_step=current_step,
                   epoch=epoch, save_n_checkpoints=None, **kwargs)  # skip built-in cleanup
    _trainer_mod.save_checkpoint = _disk_safe_save

    _orig_save_best = _trainer_mod.save_best_model
    def _disk_safe_save_best(current_loss, best_loss, config, model, out_path, *,
                             current_step, epoch, **kwargs):
        # Determine if this will actually save before deleting anything
        if isinstance(current_loss, dict) and isinstance(best_loss, dict):
            el, bl = current_loss.get("eval_loss"), best_loss.get("eval_loss")
            will_save = (el < bl) if (el is not None and bl is not None) \
                        else current_loss.get("train_loss", float("inf")) < best_loss.get("train_loss", float("inf"))
        else:
            will_save = float(current_loss) < float(best_loss)

        if will_save:
            _pre_delete_pths(out_path)

        result = _orig_save_best(current_loss, best_loss, config, model, out_path,
                                 current_step=current_step, epoch=epoch, **kwargs)

        # Replace the 5.3 GB best_model.pth copy with a symlink to save disk space
        shortcut = os.path.join(str(out_path), "best_model.pth")
        versioned = os.path.join(str(out_path), f"best_model_{current_step}.pth")
        if os.path.isfile(shortcut) and os.path.isfile(versioned) and not os.path.islink(shortcut):
            os.remove(shortcut)
            os.symlink(versioned, shortcut)
            logger.info(f"Replaced best_model.pth copy with symlink → {versioned}")

        return result
    _trainer_mod.save_best_model = _disk_safe_save_best

    logger.info("Starting fine-tuning …")
    trainer.fit()
    wandb.finish()
    logger.info("Fine-tuning complete.")
