import json

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Title
    add_md("""# 🎓 FunASR 課堂長錄音轉逐字稿與發言人分離 (MacBook Air 錄音專用優化版)

本 Notebook 專門針對以下課堂情境設計：
- **設備環境**：MacBook Air（放置於講台/講桌）錄製的長錄音（1.5 至 2.5 小時，`.m4a` 格式）。
- **音訊特性**：講師手持麥克風（音量大、清晰）；遠處提問學生無麥克風（收音微弱）。
- **語言特性**：中文為主，夾雜計算機科學、數學或醫學等英文專業名詞。
- **核心目標**：
  1. **先去噪後增益**：透過高低通濾波與頻譜去噪先壓制環境底噪，再以動態音量標準化 (`dynaudnorm`) 放大微弱學生發言，平衡師生音量。
  2. **聲紋識別與語句分割**：透過 **FunASR + CAM++** 剝離發言人（講師 vs 學生）及其發言內容。
  3. **智慧排版匯出**：匯出純文字 (`.txt`)、智慧段落 Markdown (`.md`)、影音字幕 (`.srt` / `.vtt`)、LLM 結構化總結 JSON (`.json`) 以及去噪後的音訊檔 (`.mp3` / `.wav`)。
""")

    # Step 1
    add_md("""---
## 步驟 1：環境準備與 GPU 檢測
請確保 Colab 已啟用 GPU（**執行階段 -> 變更執行階段類型 -> T4 GPU / A100**）。
此步驟會安裝 `ffmpeg`、`FunASR`、`ModelScope` 等核心依賴。
""")

    add_code("""# 1. 檢查 GPU 資源
!nvidia-smi

# 2. 確認系統 ffmpeg (Colab 已內建，若缺少則自動安裝)
!which ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)

# 3. 安裝 FunASR 及音訊依賴 (指定 decorator<5.0 與 jedi 避免 Colab 既有套件版本衝突)
!pip install -q "funasr>=1.1.0" modelscope soundfile librosa "decorator<5.0" jedi
print("✅ 環境與套件安裝完成！")""")

    # Step 2
    add_md("""---
## 步驟 2：掛載 Google Drive 與路徑設定
建議將錄音檔上傳至 Google 雲端硬碟，避免 Colab 網頁直接上傳斷線。
下方提供可直接在 Colab 介面修改的表單輸入框：
""")

    add_code("""#@title 📁 設定錄音檔路徑與輸出目錄
from google.colab import drive
import os
import glob

# 1. 掛載 Google Drive
drive.mount('/content/drive')

# 【設定參數】請填寫您錄音檔在 Google Drive 中的實際路徑
# 💡 提示：Google 預設資料夾通常是 "Colab Notebooks" (中間有空格)，而非 "Colab_Notebooks" (底線)
INPUT_AUDIO_PATH = "/content/drive/MyDrive/Colab Notebooks/Data/LinearAlgebra.m4a" #@param {type:"string"}
OUTPUT_DIR = "/content/drive/MyDrive/VoiceToText_Output" #@param {type:"string"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 檢查檔案是否存在 (若不存在自動搜尋)
if os.path.exists(INPUT_AUDIO_PATH):
    file_size_mb = os.path.getsize(INPUT_AUDIO_PATH) / (1024 * 1024)
    print(f"✅ 找到錄音檔案！大小: {file_size_mb:.2f} MB")
    print(f"檔案路徑: {INPUT_AUDIO_PATH}")
else:
    print(f"⚠️ 指定路徑不存在: {INPUT_AUDIO_PATH}")
    print("🔍 正在為您搜尋 Google Drive 中的相關檔案...")
    
    target_filename = os.path.basename(INPUT_AUDIO_PATH)
    matches = glob.glob(f"/content/drive/MyDrive/**/{target_filename}", recursive=True)
    if not matches:
        matches = glob.glob("/content/drive/MyDrive/**/*.m4a", recursive=True)
    
    if matches:
        print("\\n💡 找到以下可能的檔案，請確認是否為其中之一：")
        for idx, m in enumerate(matches[:5], 1):
            print(f"  [{idx}] {m}")
        print("\\n👉 請複製上方正確的路徑，貼回 INPUT_AUDIO_PATH 後重新執行本單元格！")
    else:
        print("\\n❌ 未在 Google Drive 找到對應音檔。請確認檔案已上傳完成，或由 Colab 左側檔案欄複製正確路徑。")

print(f"\\n結果輸出目錄: {OUTPUT_DIR}")""")

    # Step 3
    add_md("""---
## 步驟 3：音訊前處理（先濾波去噪，後動態增益）

### 💡 濾波鏈順序的最佳化設計：
若直接放大音訊，會將環境底噪與電流聲一同放大 10 餘倍，導致聲音發虛刺耳。
因此我們採用科學的處理順序：
1. **`highpass=f=80`**：濾除講台共振、敲擊低頻與冷氣嗡嗡聲。
2. **`lowpass=f=8000`**：濾除 8kHz 以上無人聲頻率的尖銳高頻雜訊。
3. **`afftdn=nf=-25`**：以 FFT 頻譜降噪先抑制底噪，提高信噪比（SNR）。
4. **`dynaudnorm=p=0.95:m=12:s=15:g=31`**：在純淨語音的基礎上，動態提升遠距微弱學生聲音（最大 12 倍），同時保護講師音量不破音。
5. **重採樣至 16kHz 單聲道 WAV**：轉為 FunASR 的標準輸入格式。
""")

    add_code("""import subprocess
import os

if not os.path.exists(INPUT_AUDIO_PATH):
    raise FileNotFoundError(f"❌ 找不到音訊檔案：{INPUT_AUDIO_PATH}\\n請先在步驟 2 設定正確的路徑！")

PROCESSED_WAV = "/content/processed_lecture_16k.wav"

# 先濾波降噪，再動態增益 (g=31 為標準高斯濾波窗大小)
ffmpeg_filter = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=p=0.95:m=12:s=15:g=31"

cmd = [
    "ffmpeg", "-y",
    "-i", INPUT_AUDIO_PATH,
    "-af", ffmpeg_filter,
    "-ar", "16000",             # 採樣率 16kHz
    "-ac", "1",                 # 單聲道
    "-c:a", "pcm_s16le",        # 16-bit PCM WAV
    PROCESSED_WAV
]

print("⏳ 正在進行音訊前處理（去噪 + 動態增益 + 格式轉換，通常需 1~3 分鐘）...")
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if result.returncode == 0:
    out_mb = os.path.getsize(PROCESSED_WAV) / (1024 * 1024)
    print(f"✅ 前處理完成！處理後音檔大小: {out_mb:.2f} MB")
else:
    print("❌ FFmpeg 處理失敗，詳細錯誤訊息如下：")
    print(result.stderr)""")

    # Step 4
    add_md("""---
## 步驟 4：設定專業領域熱詞（Hotwords）
FunASR 支援傳入 **Hotwords（熱詞清單）**，可顯著校正課堂中夾雜的英文專業名詞。
您可從下拉選單直接切換領域，或在下方欄位自由補充自訂單字：
""")

    add_code("""#@title 🏷️ 選擇熱詞預設庫與自訂補充
import re

DOMAIN_PRESETS = {
    "CS": "Python PyTorch TensorFlow Transformer Backpropagation API Docker Kubernetes GPU CUDA LLM Embedding Attention Git Vector Cache Frontend Backend SQL",
    "Math": "Matrix Vector Eigenvalue Eigenvector Determinant Derivative Integral Gradient Convex Topology Manifold Probability Distribution Poisson Gaussian",
    "Medicine": "DNA RNA PCR CRISPR Antibody Antigen CT MRI Ultrasound Pathology Symptom Diagnosis Therapy Receptor Enzyme Protein Mutation",
    "None": ""
}

SELECTED_DOMAIN = "Math" #@param ["CS", "Math", "Medicine", "None"]
CUSTOM_HOTWORDS = "Eigenvalues Invertible Span Linear Transformation Basis" #@param {type:"string"}

# 乾淨化熱詞字串 (去除多餘空白與特殊字元)
combined_raw = f"{DOMAIN_PRESETS.get(SELECTED_DOMAIN, '')} {CUSTOM_HOTWORDS}"
cleaned_hotwords = re.sub(r'\\s+', ' ', combined_raw).strip()

print(f"📌 當前啟用的熱詞清單 ({SELECTED_DOMAIN} 領域):")
print(cleaned_hotwords if cleaned_hotwords else "(未設定熱詞)")""")

    # Step 5
    add_md("""---
## 步驟 5：載入 FunASR 全管線模型
我們載入四合一完整管道：
- **`model="paraformer-zh"`**：高精度中英雙語語音辨識模型
- **`vad_model="fsmn-vad"`**：語音活動檢測，切分句子並濾除課堂靜音
- **`punc_model="ct-punc"`**：標點符號自動修復模型
- **`spk_model="cam++"`**：聲紋特徵抽取與聚類模型（區分講師與學生）
""")

    add_code("""import sys
# 💡 阻斷 torchvision 載入以避免 C++ 算子版本相容性問題 (FunASR 語音辨識與聲紋模型完全不需要視覺庫)
sys.modules['torchvision'] = None

from funasr import AutoModel
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用運算設備: {device}")

# 清理未釋放顯存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("⏳ 正在下載並初始化 FunASR 模型矩陣 (首次載入約需 1~3 分鐘)...")
model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    punc_model="ct-punc",
    punc_model_revision="v2.0.4",
    spk_model="cam++",
    spk_model_revision="v2.0.2",
    device=device
)
print("✅ 模型載入完成！準備執行辨識與發言人分離。")""")

    # Step 6
    add_md("""---
## 步驟 6：執行辨識與發言人分離
針對 1.5 ~ 2.5 小時長錄音，設定 `batch_size_s=300` 可批次分塊處理，避免 GPU 顯存溢出 (OOM)。
在 T4 GPU 上處理 2 小時音訊約耗時 5 ~ 15 分鐘。
""")

    add_code("""import time
import sys
import torch

# 💡 修復 UMAP 內部載入 torchvision 時的 C++ 算子版本不相容問題 (operator torchvision::nms does not exist)
# FunASR 語音與聲紋模型完全不需要 torchvision 視覺庫，阻斷其載入可讓 UMAP 安全優雅地略過並完成聚類
sys.modules['torchvision'] = None

if torch.cuda.is_available():
    torch.cuda.empty_cache()

start_t = time.time()
print("🎙️ 開始進行語音辨識、標點生成與發言人聚類...")

# 執行推理
generate_kwargs = {
    "input": PROCESSED_WAV,
    "batch_size_s": 300,
}
if cleaned_hotwords:
    generate_kwargs["hotword"] = cleaned_hotwords

inference_res = model.generate(**generate_kwargs)

elapsed_t = time.time() - start_t
print(f"🎉 辨識完成！總耗時: {elapsed_t / 60:.2f} 分鐘")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

raw_data = inference_res[0] if inference_res else {}
sentence_list = raw_data.get("sentence_info", [])
print(f"總計識別出 {len(sentence_list)} 個發言句子。")""")

    # Step 7
    add_md("""---
## 步驟 7：多格式智慧匯出器（TXT, Markdown, SRT, VTT, JSON, 輕量音檔）
本步驟會將辨識結果自動轉換並輸出多種格式至 Google Drive：
1. **`.txt`**：簡易純文字逐字稿。
2. **`.md`**：智慧排版筆記（同一發言人發言自動整合為段落；若停頓超過 20 秒則貼心換段）。
3. **`.srt` / `.vtt`**：標準字幕檔（含毫秒時間軸與講者標籤，具備時間戳異常防護）。
4. **`.json`**：結構化格式（附帶為 LLM 如 ChatGPT / Gemini 定制的大學課堂總結 Prompt）。
5. **去噪增益音檔**：自動備份至雲端硬碟，並額外轉成輕量 `.mp3`（約 30~50MB）方便隨身複習。
""")

    add_code("""import json
import shutil
import subprocess
from datetime import timedelta

base_filename = os.path.splitext(os.path.basename(INPUT_AUDIO_PATH))[0]
txt_path = os.path.join(OUTPUT_DIR, f"{base_filename}_transcript.txt")
md_path = os.path.join(OUTPUT_DIR, f"{base_filename}_notes.md")
srt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.srt")
vtt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.vtt")
json_path = os.path.join(OUTPUT_DIR, f"{base_filename}_structured.json")
cleaned_wav_path = os.path.join(OUTPUT_DIR, f"{base_filename}_cleaned.wav")
cleaned_mp3_path = os.path.join(OUTPUT_DIR, f"{base_filename}_cleaned.mp3")

def format_timestamp_srt(ms):
    td = timedelta(milliseconds=max(0, ms))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_timestamp_vtt(ms):
    td = timedelta(milliseconds=max(0, ms))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def format_timestamp_display(ms):
    td = timedelta(milliseconds=max(0, ms))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# 1. 輸出 TXT
with open(txt_path, "w", encoding="utf-8") as f_txt:
    for item in sentence_list:
        spk = item.get("spk", "0")
        start_str = format_timestamp_display(item.get("start", 0))
        f_txt.write(f"[{start_str}] [Speaker {spk}]: {item.get('text', '')}\\n")

# 2. 輸出 SRT 與 VTT (含時間戳防異常校正)
with open(srt_path, "w", encoding="utf-8") as f_srt, open(vtt_path, "w", encoding="utf-8") as f_vtt:
    f_vtt.write("WEBVTT\\n\\n")
    for idx, item in enumerate(sentence_list, 1):
        spk = item.get("spk", "0")
        s_ms = item.get("start", 0)
        e_ms = item.get("end", s_ms + 1000)
        if e_ms <= s_ms:
            e_ms = s_ms + 1000
            
        start_srt = format_timestamp_srt(s_ms)
        end_srt = format_timestamp_srt(e_ms)
        start_vtt = format_timestamp_vtt(s_ms)
        end_vtt = format_timestamp_vtt(e_ms)
        text = item.get("text", "")
        
        # SRT
        f_srt.write(f"{idx}\\n{start_srt} --> {end_srt}\\n[Speaker {spk}] {text}\\n\\n")
        # VTT
        f_vtt.write(f"{idx}\\n{start_vtt} --> {end_vtt}\\n<v Speaker {spk}>{text}</v>\\n\\n")

# 3. 輸出 Markdown (智慧合併同一說話者發言，停頓超過 20 秒自動分段)
with open(md_path, "w", encoding="utf-8") as f_md:
    f_md.write(f"# 課堂錄音逐字稿：{base_filename}\\n\\n")
    f_md.write(f"> 總句數：{len(sentence_list)} 句 | 辨識耗時：{elapsed_t / 60:.1f} 分鐘\\n\\n---\\n\\n")
    
    current_spk = None
    paragraph = []
    para_start = 0
    last_end = 0
    
    for item in sentence_list:
        spk = item.get("spk", "0")
        s_ms = item.get("start", 0)
        e_ms = item.get("end", s_ms)
        text = item.get("text", "")
        
        # 換說話人 或 停頓超過 20 秒時，結算前一個段落
        is_speaker_change = (spk != current_spk)
        is_long_pause = (last_end > 0 and (s_ms - last_end) > 20000)
        
        if (is_speaker_change or is_long_pause) and paragraph:
            time_tag = format_timestamp_display(para_start)
            f_md.write(f"### 🗣️ Speaker {current_spk} ({time_tag})\\n\\n" + "".join(paragraph) + "\\n\\n")
            paragraph = []
            para_start = s_ms
            
        if not paragraph:
            para_start = s_ms
            
        current_spk = spk
        last_end = e_ms
        paragraph.append(text)
        
    if paragraph:
        time_tag = format_timestamp_display(para_start)
        f_md.write(f"### 🗣️ Speaker {current_spk} ({time_tag})\\n\\n" + "".join(paragraph) + "\\n\\n")

# 4. 輸出結構化 JSON (含 LLM 總結 Prompt 模板)
spk_counts = {}
for item in sentence_list:
    spk = str(item.get("spk", "0"))
    spk_counts[spk] = spk_counts.get(spk, 0) + 1

json_payload = {
    "metadata": {
        "audio_file": os.path.basename(INPUT_AUDIO_PATH),
        "total_sentences": len(sentence_list),
        "speakers_detected": list(spk_counts.keys()),
        "speaker_sentence_distribution": spk_counts,
        "processing_time_seconds": round(elapsed_t, 2)
    },
    "llm_prompt_suggestion": (
        "你是一位專業的大學課程助教。以下為課堂逐字稿，包含了講師與學生的發言互動。\\n"
        "請根據這份逐字稿完成：\\n"
        "1. 【課程核心概念總結】：條列本日講述的主要定理、概念與推導。\\n"
        "2. 【問答互動摘要】：整理學生提出了哪些問題（注意提問者通常句子較短），講師是如何解答的。\\n"
        "3. 【重點專有名詞表】：列出課堂中提到的中英文專業術語與對應解釋。\\n\\n"
        "【逐字稿內容如下】\\n"
    ),
    "segments": [
        {
            "id": idx + 1,
            "speaker": f"Speaker {item.get('spk', '0')}",
            "start_time_seconds": round(item.get("start", 0) / 1000.0, 2),
            "end_time_seconds": round(item.get("end", 0) / 1000.0, 2),
            "text": item.get("text", "")
        }
        for idx, item in enumerate(sentence_list)
    ]
}

with open(json_path, "w", encoding="utf-8") as f_json:
    json.dump(json_payload, f_json, ensure_ascii=False, indent=2)

# 5. 備份去噪增益後的音檔至 Google Drive (含輕量化 MP3)
print("💾 正在將去噪增益後的音檔保存至 Google Drive...")
shutil.copy(PROCESSED_WAV, cleaned_wav_path)
subprocess.run(["ffmpeg", "-y", "-i", PROCESSED_WAV, "-b:a", "96k", cleaned_mp3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\\n🎉 所有成果已成功儲存至 Google Drive：")
print(f"1. 純文字檔: {txt_path}")
print(f"2. Markdown 筆記: {md_path}")
print(f"3. SRT 字幕: {srt_path}")
print(f"4. VTT 字幕: {vtt_path}")
print(f"5. 結構化 JSON: {json_path}")
print(f"6. 去噪音檔 (WAV): {cleaned_wav_path}")
print(f"7. 去噪音檔 (輕量 MP3): {cleaned_mp3_path}")""")

    # Step 8
    add_md("""---
## 步驟 8：發言人統計與逐字稿預覽
在此快速檢視發言人比例分佈，並預覽前 15 句對話。
""")

    add_code("""total_sentences = len(sentence_list)
print("📊 發言人句數統計：")
if total_sentences > 0:
    for spk, count in sorted(spk_counts.items()):
        ratio = (count / total_sentences) * 100
        role_hint = "主要講者 (推測為講師)" if ratio > 60 else "提問或次要發言者"
        print(f" - Speaker {spk}: {count} 句 ({ratio:.1f}%) -> {role_hint}")
else:
    print("⚠️ 未檢測到有效發言句子。")

print("\\n" + "="*50 + "\\n前 15 句逐字稿快速預覽：\\n" + "="*50)
for item in sentence_list[:15]:
    spk = item.get("spk", "0")
    start_s = item.get("start", 0) / 1000.0
    print(f"[{start_s:06.1f}s] [Speaker {spk}]: {item.get('text', '')}")""")

    # Step 9
    add_md("""---
## 步驟 9：（可選）下載結果與音檔回本機
您可以在此選擇將文字逐字稿壓縮包或去噪後的 MP3 音檔下載至本機電腦。
""")

    add_code("""#@title 📦 下載結果到本機
from google.colab import files
import shutil

DOWNLOAD_TRANSCRIPTS_ZIP = True #@param {type:"boolean"}
DOWNLOAD_CLEANED_MP3 = False #@param {type:"boolean"}

# 1. 下載文字/字幕/JSON 逐字稿 (打包為 ZIP，僅數百 KB)
if DOWNLOAD_TRANSCRIPTS_ZIP:
    zip_filename = f"/content/{base_filename}_transcription_results.zip"
    zip_folder = "/content/zip_temp"
    os.makedirs(zip_folder, exist_ok=True)
    
    for p in [txt_path, md_path, srt_path, vtt_path, json_path]:
        if os.path.exists(p):
            shutil.copy(p, zip_folder)
            
    shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', zip_folder)
    print("📦 正在啟動逐字稿 ZIP 壓縮包下載...")
    files.download(zip_filename)

# 2. 下載去噪增益後的 MP3 音檔 (約 30~50MB)
if DOWNLOAD_CLEANED_MP3:
    if os.path.exists(cleaned_mp3_path):
        print("🎵 正在啟動去噪 MP3 音檔下載...")
        files.download(cleaned_mp3_path)
    else:
        print("⚠️ 找不到已轉換的 MP3 檔案，請確認步驟 7 已完成執行。")""")

    # Write out
    with open("main.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("Successfully generated main.ipynb!")

if __name__ == "__main__":
    create_notebook()
