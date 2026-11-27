import glob
import os
import tempfile
import requests
import streamlit as st
from pydub import AudioSegment

st.set_page_config(page_title="ずんだもん音声スタジオ", page_icon="🌱")
st.title("🌱 ずんだもん音声スタジオ（VOICEVOX）")
st.caption("①ファイルから全文抽出 ②テキストの読み上げ ③マイクで音を拾って抽出＆お喋り")

VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")


@st.cache_resource
def load_whisper_model():
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("large", device=device)


def synthesize_zunda(
    text: str,
    speaker_id: int,
    speed: float,
    pitch: float,
    intonation: float,
    auto_zunda: bool,
):
    t = text.strip()[:300]
    if auto_zunda and t and t[-1] not in "。！？!?":
        t += "なのだ！"

    q_res = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": t, "speaker": speaker_id},
        timeout=20,
    )
    q_res.raise_for_status()
    query_data = q_res.json()
    query_data["speedScale"] = speed
    query_data["pitchScale"] = pitch
    query_data["intonationScale"] = intonation

    s_res = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query_data,
        timeout=30,
    )
    s_res.raise_for_status()
    return s_res.content, t


def transcribe_audio_bytes(audio_bytes: bytes, prefix: str = "in") -> str:
    model = load_whisper_model()
    in_path = os.path.join(tempfile.gettempdir(), f"{prefix}_raw.wav")
    out_path = os.path.join(tempfile.gettempdir(), f"{prefix}_16k.wav")
    with open(in_path, "wb") as f:
        f.write(audio_bytes)
    seg = AudioSegment.from_file(in_path).set_channels(1).set_frame_rate(16000)
    seg.export(out_path, format="wav")
    res = model.transcribe(out_path, language="ja")
    return res.get("text", "").strip()


with st.sidebar:
    st.header("ずんだもん設定")
    speaker_id = st.selectbox(
        "スタイル（話者ID）",
        options=[3, 1, 7, 5, 22, 38, 75, 76],
        format_func=lambda x: {
            3: "ノーマル（ID: 3）",
            1: "あまあま（ID: 1）",
            7: "ツンツン（ID: 7）",
            5: "セクシー（ID: 5）",
            22: "ささやき（ID: 22）",
            38: "ヒソヒソ（ID: 38）",
            75: "ヘロヘロ（ID: 75）",
            76: "なみだめ（ID: 76）",
        }.get(x, f"ID: {x}"),
        index=0,
    )
    speed = st.slider("話す速さ", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("声の高さ", -0.15, 0.15, 0.0, 0.01)
    intonation = st.slider("抑揚", 0.0, 2.0, 1.0, 0.1)
    auto_zunda = st.checkbox("語尾に「なのだ！」を自動付与", value=True)
    drive_dir = st.text_input("ドライブ音声フォルダ", value="/content/drive/MyDrive")

tab1, tab2, tab3 = st.tabs([
    "📂 1. ファイルから抽出",
    "🗣️ 2. テキストの読み上げ",
    "🎤 3. マイクで音を拾って抽出＆お喋り",
])

with tab1:
    up = st.file_uploader(
        "音声ファイルをアップロード",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
    )
    drive_files = []
    if os.path.exists(drive_dir):
        drive_files = sorted([
            p
            for p in glob.glob(os.path.join(drive_dir, "**/*"), recursive=True)
            if os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS)
        ])
    sel_drive = st.selectbox(
        "または Googleドライブ内の音声ファイル",
        options=[""] + drive_files,
        format_func=lambda p: "（選択しない）" if not p else os.path.relpath(p, drive_dir),
    )

    src_bytes = None
    src_label = None
    if up is not None:
        src_bytes = up.getvalue()
        src_label = up.name
    elif sel_drive:
        with open(sel_drive, "rb") as f:
            src_bytes = f.read()
        src_label = os.path.basename(sel_drive)

    if src_bytes is not None:
        st.audio(src_bytes)
        tmp_src = os.path.join(tempfile.gettempdir(), "extract_src")
        with open(tmp_src, "wb") as f:
            f.write(src_bytes)
        seg = AudioSegment.from_file(tmp_src)
        dur = round(len(seg) / 1000.0, 2)
        st.write(f"対象ファイル: **{src_label}**（長さ: **{dur} 秒**）")

        if st.button("ファイル全体から発話テキストを抽出 📝"):
            try:
                with st.spinner("音声全体からテキストを抽出中..."):
                    st.session_state["file_transcript"] = transcribe_audio_bytes(
                        src_bytes, "file_tr"
                    )
            except Exception as e:
                st.error(f"テキスト抽出エラー: {e}")

    if "file_transcript" in st.session_state and st.session_state["file_transcript"]:
        transcript_text = st.session_state["file_transcript"]
        st.text_area(
            "抽出したテキスト",
            value=transcript_text,
            height=140,
        )
        try:
            if st.session_state.get("auto_tts_source") != transcript_text:
                with st.spinner("抽出したテキストからずんだもん音声を生成中..."):
                    wav_bytes, spoken = synthesize_zunda(
                        transcript_text, speaker_id, speed, pitch, intonation, auto_zunda
                    )
                    st.session_state["auto_tts_wav"] = wav_bytes
                    st.session_state["auto_tts_spoken"] = spoken
                    st.session_state["auto_tts_source"] = transcript_text
            if "auto_tts_wav" in st.session_state:
                st.audio(st.session_state["auto_tts_wav"], format="audio/wav", autoplay=True)
        except Exception as e:
            st.error(f"抽出テキストの音声化エラー: {e}")

with tab2:
    text = st.text_area(
        "読み上げるテキスト",
        "こんにちは！ぼくはずんだもんなのだ！今日も元気に喋るのだ！",
        max_chars=300,
    )
    if st.button("ずんだもんに読み上げさせる 🔊") and text.strip():
        try:
            with st.spinner("ずんだもん音声を生成中..."):
                wav_bytes, spoken = synthesize_zunda(
                    text, speaker_id, speed, pitch, intonation, auto_zunda
                )
            st.session_state["tts_wav"] = wav_bytes
            st.session_state["tts_spoken"] = spoken
        except Exception as e:
            st.error(f"読み上げエラー: {e}")

    if "tts_wav" in st.session_state:
        st.audio(st.session_state["tts_wav"], format="audio/wav", autoplay=True)
        st.download_button(
            label="読み上げWAVを保存 💾",
            data=st.session_state["tts_wav"],
            file_name="zundamon_tts.wav",
            mime="audio/wav",
        )
        st.success(f"読み上げテキスト: 「{st.session_state.get('tts_spoken', '')}」")

with tab3:
    st.write("マイクで録音した音声から言葉を抽出し、ずんだもんの声でお喋りさせます。")
    mic_audio = st.audio_input("マイクで録音する 🎙️") if hasattr(st, "audio_input") else None
    mic_upload = st.file_uploader(
        "または録音済みマイク音声（マイク非対応環境用）",
        type=["wav", "mp3", "m4a", "ogg"],
        key="mic_up",
    )

    mic_bytes = None
    if mic_audio is not None:
        mic_bytes = mic_audio.getvalue()
    elif mic_upload is not None:
        mic_bytes = mic_upload.getvalue()

    if mic_bytes is not None:
        st.audio(mic_bytes)
        if st.button("マイク音声を抽出してずんだもんがお喋り 🔊"):
            try:
                with st.spinner("マイク音声から言葉を抽出中..."):
                    heard = transcribe_audio_bytes(mic_bytes, "mic_in")
                st.session_state["mic_heard"] = heard
                if not heard:
                    st.warning("音声から言葉を抽出できませんでした。もう一度はっきり話してみてください。")
                else:
                    with st.spinner("ずんだもん音声を生成中..."):
                        wav_bytes, spoken = synthesize_zunda(
                            heard, speaker_id, speed, pitch, intonation, auto_zunda
                        )
                    st.session_state["mic_zunda_wav"] = wav_bytes
                    st.session_state["mic_spoken"] = spoken
            except Exception as e:
                st.error(f"マイク処理エラー: {e}")

    if "mic_heard" in st.session_state:
        st.text_input("抽出したマイク音声テキスト", value=st.session_state["mic_heard"])
    if "mic_zunda_wav" in st.session_state:
        st.audio(st.session_state["mic_zunda_wav"], format="audio/wav", autoplay=True)
        st.success(f"ずんだもんの発話: 「{st.session_state.get('mic_spoken', '')}」")
