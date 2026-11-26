import os, re, uuid, hashlib, tempfile, requests, torch, whisper, pykakasi
import soundfile as sf, numpy as np, streamlit as st
from pydub import AudioSegment
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

os.environ["COQUI_TOS_AGREED"] = "1"

if not getattr(torch.load, "_patched", False):
    _orig_load = torch.load
    def _safe_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    _safe_load._patched = True
    torch.load = _safe_load

st.set_page_config(page_title="AIボイスチェンジャー", page_icon="🌱")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "1234")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "325A1104/katou-xtts-private")

if "authed" not in st.session_state:
    st.session_state.authed = False
if "katou_mode" not in st.session_state:
    st.session_state.katou_mode = False
if "last_rt_hash" not in st.session_state:
    st.session_state.last_rt_hash = ""

with st.sidebar:
    pw = st.text_input("アクセスコード", type="password")
    if pw == APP_PASSWORD:
        st.session_state.authed = True

if not st.session_state.authed:
    st.info("左の欄にアクセスコードを入力してください。")
    st.stop()

SECRET_WORDS = ["加藤純一", "かとうじゅんいち", "うんこちゃん", "衛門"]

def check_secret_trigger(text):
    return any(w in text for w in SECRET_WORDS)

@st.cache_resource
def get_katou_assets():
    cfg_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json", token=HF_TOKEN)
    mdl_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pth", token=HF_TOKEN)
    vcb_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab.json", token=HF_TOKEN)
    refs = []
    for i in [1, 2, 3, 4, 5, 6, 8, 10]:
        try:
            r = hf_hub_download(repo_id=HF_REPO_ID, filename=f"refs/best_{i:03d}.wav", token=HF_TOKEN)
            refs.append(r)
        except Exception:
            pass

    config = XttsConfig()
    config.load_json(cfg_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=mdl_path, vocab_path=vcb_path, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    _orig_gen = model.gpt.gpt_inference.generate
    def _no_cache_gen(*args, **kwargs):
        kwargs["use_cache"] = False
        return _orig_gen(*args, **kwargs)
    model.gpt.gpt_inference.generate = _no_cache_gen
    return model, refs

@st.cache_resource
def get_kakasi():
    return pykakasi.kakasi()

@st.cache_resource
def get_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "large-v3" if device == "cuda" else "base"
    return whisper.load_model(model_size, device=device)

def to_natural_kana(text):
    kks = get_kakasi()
    return "".join([item["hira"] for item in kks.convert(text)])

def normalize_audio_to_wav(raw_bytes, ext):
    in_tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    out_wav = os.path.join(tempfile.gettempdir(), f"norm_{uuid.uuid4().hex}.wav")
    try:
        in_tmp.write(raw_bytes)
        in_tmp.close()
        audio = AudioSegment.from_file(in_tmp.name).set_channels(1).set_frame_rate(16000)
        audio.export(out_wav, format="wav")
        return out_wav
    finally:
        if os.path.exists(in_tmp.name):
            os.remove(in_tmp.name)

def transcribe_clean(wav_path):
    res = get_whisper().transcribe(
        wav_path, language="ja", beam_size=1,
        condition_on_previous_text=False, no_speech_threshold=0.6, compression_ratio_threshold=1.8
    )
    return res["text"].strip()

def synthesize_zunda_direct(text):
    text = text.strip()[:300]
    if not text:
        return None
    if text[-1] not in "。！？!?":
        text += "なのだ！"
    q_res = requests.post("http://127.0.0.1:50021/audio_query", params={"text": text, "speaker": 3}, timeout=15)
    q_res.raise_for_status()
    s_res = requests.post("http://127.0.0.1:50021/synthesis", params={"speaker": 3}, json=q_res.json(), timeout=30)
    s_res.raise_for_status()
    return s_res.content

def synthesize_katou_bytes(text, ref_count, temp, use_kana):
    model, all_refs = get_katou_assets()
    refs = all_refs[:ref_count]
    text = text.strip()[:300]
    if not text:
        return None
    if text[-1] not in "。！？!?":
        text += "。"
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=refs, gpt_cond_len=12, max_ref_length=30
    )
    parts = [s.strip() for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
    wavs = []
    for p in parts:
        inp = to_natural_kana(p) if use_kana else p
        out = model.inference(
            text=inp, language="ja",
            gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding,
            temperature=temp, length_penalty=1.0, repetition_penalty=7.0,
            top_k=50, top_p=0.85, speed=1.0, enable_text_splitting=False
        )
        w = out.get("wav")
        if w is not None and len(w) > 0:
            wav_np = np.array(w, dtype=np.float32)
            peak = np.max(np.abs(wav_np))
            if peak > 0:
                wav_np = wav_np / peak * 0.92
            wavs.append(wav_np)
    if not wavs:
        return None
    out_path = os.path.join(tempfile.gettempdir(), f"katou_{uuid.uuid4().hex}.wav")
    try:
        sf.write(out_path, np.concatenate(wavs), 24000)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_voice_generation(text, ref_count, temp, use_kana, auto_play=True):
    if not st.session_state.katou_mode and check_secret_trigger(text):
        st.session_state.katou_mode = True
        st.balloons()
        st.toast("🔥 隠しコマンド検知！『加藤純一モード』が解放されました！！", icon="🔥")

    if st.session_state.katou_mode:
        audio_bytes = synthesize_katou_bytes(text, ref_count, temp, use_kana)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav", autoplay=auto_play)
            st.caption("🔥 加藤純一モード（35epoch学習済み専用AIモデル）")
    else:
        audio_bytes = synthesize_zunda_direct(text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav", autoplay=auto_play)
            st.caption("🌱 VOICEVOX:ずんだもん（公式エンジン直接合成・即時再生）")

if st.session_state.katou_mode:
    st.title("🔥 AIボイスチェンジャー【加藤純一モード解放中】")
    st.success("⚡ ファインチューニング済み専用モデル（35epoch）が有効化されています！")
else:
    st.title("🌱 ずんだもん AIボイスチェンジャーなのだ！")
    st.caption("マイクで喋るだけでリアルタイムにずんだもんの声で喋るのだ！（クレジット: VOICEVOX:ずんだもん）")

with st.sidebar:
    st.header("⚙️ 声質・抑揚調整")
    ref_mode = st.radio("参照音声セット", ["標準（5件）", "高密度（8件・再現重視）"], index=1)
    ref_count = 5 if "5件" in ref_mode else 8
    temp = st.slider("抑揚の豊かさ（0.55=安定 / 0.70=感情豊か）", 0.50, 0.80, 0.68, 0.02)
    use_kana = st.checkbox("ひらがな自動変換（棒読み防止・推奨）", value=True)
    if st.session_state.katou_mode:
        st.divider()
        if st.button("🌱 ずんだもんモードに戻す"):
            st.session_state.katou_mode = False
            st.rerun()

t_rt, t1, t2 = st.tabs(["⚡ リアルタイム変換（マイク即再生）", "✍️ テキスト入力", "📁 音声ファイル"])

with t_rt:
    st.write("🎙️ 下のマイクボタンを押して喋り終えると、**自動で文字起こし＆即座に自動再生**します。")
    rt_mic = st.audio_input("リアルタイムマイク入力", key="rt_mic_input")
    if rt_mic is not None:
        raw_bytes = rt_mic.getvalue()
        cur_hash = hashlib.md5(raw_bytes).hexdigest()
        if cur_hash != st.session_state.last_rt_hash:
            st.session_state.last_rt_hash = cur_hash
            try:
                with st.spinner("⚡ リアルタイム音声認識＆発話中..."):
                    norm_wav = normalize_audio_to_wav(raw_bytes, ".wav")
                    try:
                        recognized = transcribe_clean(norm_wav)
                    finally:
                        if os.path.exists(norm_wav):
                            os.remove(norm_wav)
                    st.session_state["last_rt_text"] = recognized
                    if recognized:
                        st.info(f"🗣️ 認識結果: **{recognized}**")
                        run_voice_generation(recognized, ref_count, temp, use_kana, auto_play=True)
                    else:
                        st.warning("音声が聞き取れませんでした。もう一度話してみてください。")
            except Exception as e:
                st.error(f"リアルタイム変換エラー: {e}")
        elif "last_rt_text" in st.session_state and st.session_state["last_rt_text"]:
            st.info(f"🗣️ 直前の認識結果: **{st.session_state['last_rt_text']}**")
            if st.button("もう一度同じ内容で再生 🔁", key="replay_rt"):
                run_voice_generation(st.session_state["last_rt_text"], ref_count, temp, use_kana, auto_play=True)

with t1:
    default_txt = (
        "なんだ、全然痛くね？つって。は？こんだけ食らってるのに、なんだこれ……なんで俺は死なねえんだ？"
        if st.session_state.katou_mode
        else "こんにちは！ぼくはずんだもんなのだ！今日も元気に喋るのだ！"
    )
    txt = st.text_area("喋らせるセリフ", default_txt, max_chars=300)
    if st.button("喋らせる 🔊", key="btn_txt") and txt.strip():
        try:
            with st.spinner("音声生成中..."):
                run_voice_generation(txt, ref_count, temp, use_kana, auto_play=True)
        except Exception as e:
            st.error(f"音声生成エラー: {e}")

with t2:
    up = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "m4a"])
    if up and st.button("1. 文字起こしを実行 📝", key="btn_file_tr"):
        try:
            with st.spinner("音声を文字起こし中..."):
                ext = os.path.splitext(up.name)[1].lower() or ".wav"
                norm_wav = normalize_audio_to_wav(up.getvalue(), ext)
                try:
                    st.session_state["file_text"] = transcribe_clean(norm_wav)
                finally:
                    if os.path.exists(norm_wav):
                        os.remove(norm_wav)
        except Exception as e:
            st.error(f"文字起こしエラー: {e}")
    if "file_text" in st.session_state:
        edited_file_txt = st.text_area("認識テキスト（必要に応じて修正できます）", st.session_state["file_text"], key="edit_file")
        if st.button("2. この内容で喋らせる 🔊", key="btn_file_sp") and edited_file_txt.strip():
            with st.spinner("音声生成中..."):
                run_voice_generation(edited_file_txt, ref_count, temp, use_kana, auto_play=True)
