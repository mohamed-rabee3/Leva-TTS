"""
🎙️  Leva-TTS  ·  Gradio Demo

Run:  python app.py
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
CHECKPOINT_DIR   = "checkpoints"
REFERENCES_JSON  = "reference_audios/references.json"
DEFAULT_SPEAKER  = "Mohamed"
DEVICE           = "cuda"
SERVER_PORT      = 7860
SHARE            = False
# ╚══════════════════════════════════════════════════════════════════════════╝

import json
import time
from pathlib import Path

import numpy as np

# ── Model singleton (shared across ALL import contexts of this process) ───────
import leva_tts._model as _model_state

print("🔄 Loading model …", flush=True)
try:
    _model_state.load(checkpoint_dir=CHECKPOINT_DIR, device=DEVICE)
    print("✅ Model ready.", flush=True)
except Exception as _load_err:
    print(f"❌ Model load FAILED: {_load_err}", flush=True)
    import traceback; traceback.print_exc()


# ── Reference data ────────────────────────────────────────────────────────────
def load_refs() -> dict:
    data = json.loads(Path(REFERENCES_JSON).read_text(encoding="utf-8"))
    return {Path(r["audio_path"]).stem: r
            for r in data if Path(r["audio_path"]).exists()}

_REFS = load_refs()

# Pre-warm speaker conditioning cache
if _model_state._READY:
    print(f"⚡ Pre-caching {len(_REFS)} speaker conditionings …", flush=True)
    for _name, _ref in _REFS.items():
        try:
            _model_state.get_conditioning(_ref["audio_path"])
        except Exception as _e:
            print(f"  ⚠️  {_name}: {_e}", flush=True)
    print("✅ All conditionings cached.", flush=True)


# ── Text processing ───────────────────────────────────────────────────────────
def process_text(text: str) -> str:
    from leva_tts.text.processor import TextProcessor
    return TextProcessor().process(text)

def split_text(text: str, max_chars: int = 180) -> list:
    """
    Split long text into chunks under the XTTS token limit (≈400 tokens).
    Splits on sentence boundaries first, then commas if a sentence is too long.
    """
    import re
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    # Split on sentence enders (Arabic + Latin)
    sentences = re.split(r"(?<=[.؟!؛])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) > max_chars:
                # Split a long sentence on commas
                parts = re.split(r"(?<=[،,])\s+", s)
                cur2 = ""
                for prt in parts:
                    prt = prt.strip()
                    if len(cur2) + len(prt) + 1 <= max_chars:
                        cur2 = (cur2 + " " + prt).strip()
                    else:
                        if cur2:
                            chunks.append(cur2)
                        # Hard-split if still too long
                        while len(prt) > max_chars:
                            chunks.append(prt[:max_chars])
                            prt = prt[max_chars:]
                        cur2 = prt
                cur = cur2
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return [ch for ch in chunks if ch.strip()]



# ── Synthesis ─────────────────────────────────────────────────────────────────
SR = 24_000


def _trim_silence(wav, top_db: int = 28):
    """Trim leading/trailing silence to reduce breath gaps between segments."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        return trimmed if len(trimmed) > 32 else wav
    except Exception:
        return wav


# Default generation parameters (tuned for natural Levantine speech)
DEFAULT_GEN = {
    "temperature":        0.65,
    "length_penalty":     1.0,
    "repetition_penalty": 5.0,   # higher → fewer repeated tokens / breath artifacts
    "top_k":              50,
    "top_p":              0.85,
    "speed":              1.0,
}


def synth_batch(text: str, ref_wav: str, language: str, gen: dict | None = None):
    import torch
    gen = {**DEFAULT_GEN, **(gen or {})}
    gen["repetition_penalty"] = max(float(gen["repetition_penalty"]), 1.0)
    gen["length_penalty"]     = max(float(gen["length_penalty"]), 0.1)
    mdl, cfg = _model_state.get()
    processed  = process_text(text)
    chunks_txt = split_text(processed)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0   = time.perf_counter()
    wavs = []
    for ct in chunks_txt:
        out = mdl.synthesize(
            ct, cfg, speaker_wav=[ref_wav], language=language, gpt_cond_len=3,
            temperature        = gen["temperature"],
            length_penalty     = gen["length_penalty"],
            repetition_penalty = gen["repetition_penalty"],
            top_k              = gen["top_k"],
            top_p              = gen["top_p"],
        )
        seg = _trim_silence(np.array(out["wav"], dtype=np.float32))
        wavs.append(seg)
    wall = time.perf_counter() - t0
    wav  = np.concatenate(wavs) if wavs else np.zeros(1, np.float32)
    dur  = len(wav) / SR
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    m    = (f"⏱️ {wall*1000:.0f} ms  |  🎚️ RTF {wall/dur:.3f}  |  "
            f"🕐 {dur:.1f}s  |  💾 {vram:.2f}GB  |  {len(chunks_txt)} segs")
    return (SR, wav), processed, m


def synth_stream(text: str, ref_wav: str, language: str, gen: dict | None = None):
    """
    Streaming generator that yields CUMULATIVE audio after each segment so the
    waveform display grows progressively and the first segment is playable fast.
    Silence is trimmed between segments to minimise the breath/gap artifact.
    Yields: (audio_tuple, processed_text, status_str)
    """
    import torch
    gen = {**DEFAULT_GEN, **(gen or {})}
    gen["repetition_penalty"] = max(float(gen["repetition_penalty"]), 1.0)
    gen["length_penalty"]     = max(float(gen["length_penalty"]), 0.1)
    mdl, _ = _model_state.get()
    processed  = process_text(text)
    gpt_cond, spk_emb = _model_state.get_conditioning(ref_wav)
    chunks_txt = split_text(processed)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter(); first = True; ttfa = 0.0
    all_segs = []
    try:
        for seg_idx, seg in enumerate(chunks_txt, 1):
            seg_chunks = []
            for chunk in mdl.inference_stream(
                seg, language,
                gpt_cond_latent  = gpt_cond,
                speaker_embedding = spk_emb,
                stream_chunk_size = 20,
                temperature        = gen["temperature"],
                length_penalty     = gen["length_penalty"],
                repetition_penalty = gen["repetition_penalty"],
                top_k              = gen["top_k"],
                top_p              = gen["top_p"],
                speed              = gen["speed"],
            ):
                arr = chunk.squeeze().cpu().numpy().astype(np.float32)
                if first:
                    ttfa = (time.perf_counter() - t0) * 1000; first = False
                seg_chunks.append(arr)
            seg_wav  = _trim_silence(np.concatenate(seg_chunks)) if seg_chunks else np.zeros(1, np.float32)
            all_segs.append(seg_wav)
            combined = np.concatenate(all_segs)
            yield ((SR, combined), processed,
                   f"⚡ TTFA {ttfa:.0f} ms  |  seg {seg_idx}/{len(chunks_txt)}"
                   f"  |  {len(combined)/SR:.1f}s")
        wall = time.perf_counter() - t0
        final = np.concatenate(all_segs) if all_segs else np.zeros(1, np.float32)
        dur   = len(final) / SR
        vram  = (torch.cuda.max_memory_allocated() / 1e9
                 if torch.cuda.is_available() else 0)
        yield ((SR, final), processed,
               f"✅ TTFA {ttfa:.0f} ms  |  RTF {wall/max(dur,1e-9):.3f}  |  "
               f"{dur:.1f}s  |  💾 {vram:.2f}GB  |  {len(chunks_txt)} segs")
    except Exception as exc:
        audio, proc, m = synth_batch(text, ref_wav, language, gen)
        yield audio, proc, f"⚠️ stream→batch: {m}"



def synth_stream_chunks(text: str, ref_wav: str, language: str, gen: dict | None = None):
    """Yield INDIVIDUAL chunks for the streaming=True HTML5 player (no waveform)."""
    import torch
    gen = {**DEFAULT_GEN, **(gen or {})}
    gen["repetition_penalty"] = max(float(gen["repetition_penalty"]), 1.0)
    gen["length_penalty"]     = max(float(gen["length_penalty"]), 0.1)
    mdl, _ = _model_state.get()
    processed = process_text(text)
    gpt_cond, spk_emb = _model_state.get_conditioning(ref_wav)
    chunks_txt = split_text(processed)
    t0 = time.perf_counter(); first = True; ttfa = 0.0; total = 0; n = 0
    try:
        for seg_idx, seg in enumerate(chunks_txt, 1):
            for chunk in mdl.inference_stream(
                seg, language,
                gpt_cond_latent=gpt_cond, speaker_embedding=spk_emb,
                stream_chunk_size=20,
                temperature=gen["temperature"], length_penalty=gen["length_penalty"],
                repetition_penalty=gen["repetition_penalty"],
                top_k=gen["top_k"], top_p=gen["top_p"], speed=gen["speed"],
            ):
                arr = chunk.squeeze().cpu().numpy().astype(np.float32)
                if first:
                    ttfa = (time.perf_counter() - t0) * 1000; first = False
                n += 1; total += len(arr)
                yield ((SR, arr), processed,
                       f"⚡ TTFA {ttfa:.0f} ms  |  seg {seg_idx}/{len(chunks_txt)}  |  chunk {n}  |  {total/SR:.1f}s")
        wall = time.perf_counter() - t0; dur = total / SR if total else 0
        vram = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0
        yield (None, processed, f"✅ TTFA {ttfa:.0f} ms  |  RTF {wall/max(dur,1e-9):.3f}  |  {dur:.1f}s  |  💾 {vram:.2f}GB")
    except Exception:
        audio, proc, m = synth_batch(text, ref_wav, language, gen)
        yield audio, proc, f"⚠️ stream→batch: {m}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_demo():
    import gradio as gr

    # Shared waveform styling so the generated-speech player matches the reference
    WF = gr.WaveformOptions(
        waveform_color          = "#06b6d4",
        waveform_progress_color = "#22d3ee",
        show_controls           = False,
        sample_rate             = SR,
    )

    speaker_names = list(_REFS.keys())
    gender_icons  = {"male": "👨", "female": "👩"}
    spk_labels    = [
        f"{gender_icons.get(_REFS[n].get('gender', ''), '🎤')} {n}"
        for n in speaker_names
    ]
    label_to_name = dict(zip(spk_labels, speaker_names))

    def _gen_param_controls():
        """Build a collapsible set of XTTS generation-parameter sliders."""
        with gr.Accordion("⚙️ Advanced generation settings", open=False):
            temp = gr.Slider(0.30, 1.00, value=DEFAULT_GEN["temperature"],
                             step=0.05, label="Temperature",
                             info="Lower = more stable, higher = more expressive")
            rep  = gr.Slider(1.0, 10.0, value=DEFAULT_GEN["repetition_penalty"],
                             step=0.5, label="Repetition penalty",
                             info="Higher reduces repeated sounds / breath artifacts")
            topp = gr.Slider(0.50, 1.00, value=DEFAULT_GEN["top_p"],
                             step=0.05, label="Top-p")
            topk = gr.Slider(0, 100, value=DEFAULT_GEN["top_k"],
                             step=1, label="Top-k")
            spd  = gr.Slider(0.70, 1.50, value=DEFAULT_GEN["speed"],
                             step=0.05, label="Speed",
                             info="Streaming mode only (<1 slower, >1 faster)")
            lenp = gr.Slider(0.50, 2.00, value=DEFAULT_GEN["length_penalty"],
                             step=0.1, label="Length penalty")
        return [temp, rep, topp, topk, spd, lenp]


    EXAMPLES = [
        ["كيفك اليوم؟ إنت شو عم تعمل هَلَّق؟ بِدِّي أحكيلك عن الـ meeting اللي كان اليوم!",
         "👨 Badr", "ar"],
        ["هَلَّق أنا عم أشتغل على the new project. مِشْ بعرف ليش الـ system مِشْ شغّال on the server.",
         "👩 Amina", "ar"],
        ["والله the weather today كتير حلو، بِدِّي أطلع برا وأتمشى شوي.",
         "👩 Fatma", "ar"],
        ["بِدِّي أحكيلك عن the meeting. الـ feedback كان positive بس لازم نعمل more improvements.",
         "👨 Mohamed", "ar"],
        ["Hello, how are you doing today? I hope everything is going well with the new project.",
         "👩 Mona", "en"],
        ["لازم تراجع the report قبل بكرا أكيد. في كتير details لازم تضيفها.",
         "👨 Saad", "ar"],
        ["يلا نروح نشوف الفلم الجديد الليلة، سمعت إنه كتير حلو!",
         "👩 Haneen", "ar"],
    ]

    CSS = """
    .ref-box { border:1px solid #2d3748; border-radius:8px; padding:6px; }
    .metric  { font-family:monospace; font-size:.87em; background:#111827;
               padding:6px 10px; border-radius:6px; color:#a3e635; }
    .proc    { background:#0f172a; padding:6px 10px; border-radius:6px;
               font-size:.93em; direction:rtl; min-height:36px; }
    footer   { display:none !important }
    """

    model_ok = _model_state._READY
    header   = (
        "# 🎙️ Leva-TTS — Levantine Arabic / English TTS\n"
        "Fine-tuned XTTS-v2 · 10 speakers · Streaming · Code-switching  "
        + ("✅ Model ready" if model_ok else "❌ Model not loaded — check console")
    )

    with gr.Blocks(
        title   = "🎙️ Leva-TTS",
        theme   = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
        css     = CSS,
    ) as demo:

        gr.Markdown(header)

        with gr.Tabs():

            # ── Tab 1: Built-in speakers ──────────────────────────────────────
            with gr.Tab("🎤 Built-in Speakers"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("**Speaker**")
                        spk_dd  = gr.Dropdown(
                            choices=spk_labels, value=spk_labels[0],
                            label="Choose a speaker")
                        ref_aud = gr.Audio(
                            label="🔊 Reference audio",
                            interactive=False, elem_classes=["ref-box"],
                            waveform_options=WF)
                        ref_txt = gr.Textbox(
                            label="Reference transcript",
                            interactive=False, max_lines=3, rtl=True)
                        gr.Markdown("**Settings**")
                        lang_dd = gr.Dropdown(["ar", "en"], value="ar",
                                               label="Language")
                        mode_rd = gr.Radio(
                            ["🎵 Batch", "⚡ Streaming"],
                            value="🎵 Batch", label="Mode",
                            info="Streaming = fast HTML5 player; Batch = waveform")

                    with gr.Column(scale=2):
                        gr.Markdown("**Text Input**")
                        text_in = gr.Textbox(
                            label="Input (Arabic / English / mixed)",
                            placeholder="هَلَّق أنا عم أشتغل على the project اللي حكيتلك عنه.",
                            lines=4, rtl=True)
                        with gr.Row():
                            btn_synth = gr.Button("▶ Synthesize", variant="primary",  scale=3)
                            btn_proc  = gr.Button("🔤 Preview",   variant="secondary", scale=1)
                        proc_tb = gr.Textbox(
                            label="✨ Processed text (what XTTS-v2 receives)",
                            interactive=False, lines=2,
                            elem_classes=["proc"], rtl=True)
                        # Streaming player (HTML5) — visible in Streaming mode
                        aud_stream = gr.Audio(
                            label="🎵 Generated speech (streaming)",
                            type="numpy", autoplay=True, streaming=True,
                            visible=False)
                        # Waveform player — visible in Batch mode (matches reference)
                        aud_out = gr.Audio(
                            label="🎵 Generated speech (24 kHz)",
                            type="numpy", autoplay=True,
                            waveform_options=WF, elem_classes=["ref-box"],
                            visible=True)
                        met_tb  = gr.Textbox(
                            label="📊 Metrics", interactive=False,
                            lines=1, elem_classes=["metric"])

                # Full-width advanced settings (below, like Examples)
                gen1 = _gen_param_controls()

                gr.Markdown("---\n**💡 Examples — click any row**")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[text_in, spk_dd, lang_dd],
                    label="", examples_per_page=5)

            # ── Tab 2: Zero-shot ──────────────────────────────────────────────
            with gr.Tab("🎙️ Zero-Shot (Custom Speaker)"):
                gr.Markdown("""
### Clone any voice with a 3–10 s reference clip
Upload a clean audio recording — the model will synthesize in that voice.
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        zs_ref  = gr.Audio(
                            label="📁 Reference audio",
                            type="filepath",
                            sources=["upload", "microphone"])
                        zs_lang = gr.Dropdown(["ar", "en"], value="ar",
                                               label="Language")
                        zs_mode = gr.Radio(
                            ["🎵 Batch", "⚡ Streaming"],
                            value="🎵 Batch", label="Mode")
                    with gr.Column(scale=2):
                        zs_text = gr.Textbox(
                            label="Text",
                            placeholder="هَلَّق أنا عم أشتغل على the project...",
                            lines=5, rtl=True)
                        with gr.Row():
                            zs_btn  = gr.Button("▶ Synthesize (zero-shot)",
                                                variant="primary",  scale=3)
                            zs_prev = gr.Button("🔤 Preview",
                                                variant="secondary", scale=1)
                        zs_proc = gr.Textbox(
                            label="✨ Processed text",
                            interactive=False, lines=2,
                            elem_classes=["proc"], rtl=True)
                        zs_stream = gr.Audio(
                            label="🎵 Generated speech (streaming)",
                            type="numpy", autoplay=True, streaming=True,
                            visible=False)
                        zs_aud  = gr.Audio(
                            label="🎵 Generated speech",
                            type="numpy", autoplay=True,
                            waveform_options=WF, elem_classes=["ref-box"],
                            visible=True)
                        zs_met  = gr.Textbox(
                            label="📊 Metrics", interactive=False,
                            lines=1, elem_classes=["metric"])

                # Full-width advanced settings (below, like tab 1)
                gen2 = _gen_param_controls()

        gr.Markdown(
            "---\n"
            "🌿 **Levantine:** ق→/ʔ/ · ج→/ʒ/ · *il-* article · "
            "[Dataset](https://huggingface.co/datasets/mohammedaly22/lahgtna-levantine-tts) · "
            "[Model](https://huggingface.co/mohammedaly22/leva-tts)")

        # ── Event handlers ────────────────────────────────────────────────────
        def on_spk(label):
            name = label_to_name.get(label, label)
            r    = _REFS.get(name, {})
            return r.get("audio_path"), r.get("reference_text", "")

        spk_dd.change(on_spk, inputs=[spk_dd], outputs=[ref_aud, ref_txt])

        def _toggle_players(mode):
            import gradio as gr
            streaming = "Streaming" in mode
            return gr.update(visible=streaming), gr.update(visible=not streaming)

        mode_rd.change(_toggle_players, inputs=[mode_rd], outputs=[aud_stream, aud_out])
        zs_mode.change(_toggle_players, inputs=[zs_mode], outputs=[zs_stream, zs_aud])

        btn_proc.click(
            fn=lambda t: process_text(t) if t.strip() else "",
            inputs=[text_in], outputs=[proc_tb])

        def do_synth(text, spk_label, lang, mode, temp, rep, topp, topk, spd, lenp):
            if not text.strip():
                yield None, None, "", "⚠️ Enter text"; return
            if not _model_state._READY:
                yield None, None, text, f"❌ Model not loaded: {_model_state._ERR}"; return
            name = label_to_name.get(spk_label, spk_label)
            ref  = _REFS.get(name, {}).get("audio_path")
            if not ref or not Path(ref).exists():
                yield None, None, text, "❌ Reference audio not found"; return
            gen = {"temperature": float(temp), "repetition_penalty": float(rep),
                   "top_p": float(topp), "top_k": int(topk),
                   "speed": float(spd), "length_penalty": float(lenp)}
            try:
                if "Streaming" in mode:
                    for audio, proc, m in synth_stream_chunks(text, ref, lang, gen):
                        yield audio, None, proc, m
                else:
                    audio, proc, m = synth_batch(text, ref, lang, gen)
                    yield None, audio, proc, m
            except Exception as e:
                import traceback; traceback.print_exc()
                yield None, None, text, f"❌ {e}"

        btn_synth.click(
            fn=do_synth,
            inputs=[text_in, spk_dd, lang_dd, mode_rd] + gen1,
            outputs=[aud_stream, aud_out, proc_tb, met_tb])

        zs_prev.click(
            fn=lambda t: process_text(t) if t.strip() else "",
            inputs=[zs_text], outputs=[zs_proc])

        def do_zs(text, ref_wav, lang, mode, temp, rep, topp, topk, spd, lenp):
            if not text.strip():
                yield None, None, "", "⚠️ Enter text"; return
            if not _model_state._READY:
                yield None, None, text, f"❌ Model not loaded: {_model_state._ERR}"; return
            if not ref_wav or not Path(ref_wav).exists():
                yield None, None, text, "❌ Upload a reference audio"; return
            gen = {"temperature": float(temp), "repetition_penalty": float(rep),
                   "top_p": float(topp), "top_k": int(topk),
                   "speed": float(spd), "length_penalty": float(lenp)}
            try:
                if "Streaming" in mode:
                    for audio, proc, m in synth_stream_chunks(text, ref_wav, lang, gen):
                        yield audio, None, proc, m
                else:
                    audio, proc, m = synth_batch(text, ref_wav, lang, gen)
                    yield None, audio, proc, m
            except Exception as e:
                import traceback; traceback.print_exc()
                yield None, None, text, f"❌ {e}"

        zs_btn.click(
            fn=do_zs,
            inputs=[zs_text, zs_ref, zs_lang, zs_mode] + gen2,
            outputs=[zs_stream, zs_aud, zs_proc, zs_met])

        demo.load(on_spk, inputs=[spk_dd], outputs=[ref_aud, ref_txt])

    return demo


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    demo = build_demo()
    print(f"\n  🌐 Open: http://localhost:{SERVER_PORT}", flush=True)
    demo.launch(
        server_port = SERVER_PORT,
        share       = SHARE,
        server_name = "0.0.0.0",
        show_error  = True,
    )
