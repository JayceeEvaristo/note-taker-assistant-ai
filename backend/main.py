from fastapi import FastAPI, File, UploadFile
import shutil, os, whisper, subprocess
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from faster_whisper import WhisperModel


import time

app = FastAPI()

# transcribing model
# model = whisper.load_model("base")
model = WhisperModel("base", device="cpu", compute_type="int8")

summarizer_model = "sshleifer/distilbart-cnn-12-6"
summarizer_tokenizer = "sshleifer/distilbart-cnn-12-6"

summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, device=-1)

max_worker = multiprocessing.cpu_count()

def summarize_text(text):
    return summarizer(text, min_length=5, max_length=500, do_sample=False)

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        current_chunk.append(word)
        token_count += 1
        if token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            token_count = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(chunk: str):
    """Summarize a single text chunk into bullet points."""
    prompt = f"Summarize the following text into important bullet points:\n\n{chunk}"
    result = summarizer(prompt, max_length=150, min_length=30, do_sample=False)
    text = result[0]["summary_text"]

    # Ensure bullet formatting
    bullets = [f"• {line.strip()}" for line in text.split("\n") if line.strip()]
    return "\n".join(bullets)


def convert_file(file: UploadFile):
    if not file:
        raise ValueError("No file received")
    
    print(f"Received file: {file.filename}")

    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = os.path.splitext(temp_file)[1].lower()
    if ext == ".mp4":
        audio_file = f"{os.path.splitext(temp_file)[0]}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        os.remove(temp_file)  # cleanup original
        return audio_file
    elif ext == ".mp3":
        return temp_file
    else:
        os.remove(temp_file)
        raise ValueError("Unsupported file format")

@app.post("/start/")
async def transcribe(file: UploadFile = File(...)):
    start = time.time()
    if not file:
        return {"error": "No file received"}

    audio_file = convert_file(file)  # returns .mp3 path

    try:
        # Step 1: Transcribe
        print("Starting transcription...")
        t1 = time.time()
        segments, info = model.transcribe(audio_file)
        transcript_text = " ".join([seg.text for seg in segments])
        t2 = time.time()
        print(f"Transcription done in {t2 - t1:.2f} sec")

        # Step 2: Chunking
        chunks = chunk_text(transcript_text)

        # Step 3: Parallel summarization
        print("Starting summarization...")
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            summaries = list(executor.map(summarize_chunk, chunks))
        final_summary = " ".join(summaries)

        end = time.time()
        print(f"Total time: {end - start:.2f} sec")

        return {
            "transcript": transcript_text,
            "summary": final_summary
        }

    finally:
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

