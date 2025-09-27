from fastapi import FastAPI, File, UploadFile
import shutil, os, subprocess, time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from transcribe import transcribe_file
import itertools

app = FastAPI()

# ---------------------------
# 📌 Model loading (GPU aware)
# ---------------------------

# Whisper (use int8 float16 mix to fit GPU comfortably)
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Summarizer (BART)
s_model = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(s_model)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(s_model)
summarizer = pipeline(
    "summarization",
    model=summarizer_model,
    tokenizer=summarizer_tokenizer,
    device=0
)

# Note taker (FLAN-T5)
n_model = "google/flan-t5-base"
n_tokenizer = AutoTokenizer.from_pretrained(n_model)
note_taker = pipeline(
    "text2text-generation",
    model=n_model,
    tokenizer=n_tokenizer,
    device=0
)

# ---------------------------
# 📌 Helpers
# ---------------------------

def chunk_text_by_tokens(text: str, tokenizer, max_tokens=None, stride=50):
    """Chunk transcript safely by tokens, not words."""
    if max_tokens is None:
        max_tokens = getattr(tokenizer, "model_max_length", 512)
        if max_tokens > 4096:  # some return sentinel like 1e30
            max_tokens = 1024

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks, step = [], max_tokens - stride
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + max_tokens]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if start + max_tokens >= len(token_ids):
            break
    return chunks


def batch_iterable(iterable, bsize):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, bsize))
        if not batch:
            break
        yield batch


def convert_file(file: UploadFile):
    """Convert uploaded file to mp3 audio."""
    if not file:
        raise ValueError("No file received")

    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = os.path.splitext(temp_file)[1].lower()
    if ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        audio_file = f"{os.path.splitext(temp_file)[0]}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        os.remove(temp_file)
        return audio_file
    elif ext in [".mp3", ".wav", ".ogg", ".flac"]:
        return temp_file
    else:
        os.remove(temp_file)
        raise ValueError("Unsupported file format")


# ---------------------------
# 📌 FastAPI Endpoint
# ---------------------------

@app.post("/start/")
async def transcribe(file: UploadFile = File(...)):
    start = time.time()
    audio_file = convert_file(file)

    try:
        # 1️⃣ Transcribe
        print("Starting transcription...")
        transcript_text = transcribe_file(audio_file)

        # 2️⃣ Chunk text (safe < 512 tokens for BART)
        chunks = chunk_text_by_tokens(transcript_text, summarizer_tokenizer, max_tokens=500, stride=50)

        # 3️⃣ Summarization (batch on GPU)
        print("Summarizing...")
        summaries = []
        for batch in batch_iterable(chunks, bsize=4):  # adjust bsize depending on VRAM
            outs = summarizer(batch, truncation=True, max_length=180, min_length=20, do_sample=False)
            for out in outs:
                text = out.get("summary_text", "").strip()
                if text:
                    summaries.append("\n".join(f"• {line.strip()}" for line in text.split("\n") if line.strip()))
        final_summary = "\n\n".join(summaries)

        # 4️⃣ Notes (batch on GPU)
        print("Generating notes...")
        notes = []
        for batch in batch_iterable(chunks, bsize=4):
            prompts = [
                f"""Extract key points as clear, concise bullet notes.
                - Focus only on facts, decisions, and action items
                - No filler or repeated meta-instructions

                Text:
                {ch}""" for ch in batch
            ]
            outs = note_taker(prompts, truncation=True, max_length=180, min_length=15, do_sample=False)
            for out in outs:
                text = out.get("generated_text", "").strip()
                if text:
                    notes.append("\n".join(f"• {line.strip()}" for line in text.split("\n") if line.strip()))
        final_notes = "\n\n".join(notes)

        duration = time.time() - start
        print(f"Done in {duration:.2f} sec")

    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

    return {
        "transcript": transcript_text,
        "summary": final_summary,
        "notes": final_notes,
        "time": duration
    }
