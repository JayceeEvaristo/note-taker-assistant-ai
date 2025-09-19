from fastapi import FastAPI, File, UploadFile
import shutil, os, whisper, subprocess
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from faster_whisper import WhisperModel

from transcribe import transcribe_file


import time

app = FastAPI()

# transcribing model
# model = whisper.load_model("base")
model = WhisperModel("base", device="cpu", compute_type="int8")

s_model = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(s_model)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(s_model)

n_model = "google/flan-t5-base"
n_tokenizer = AutoTokenizer.from_pretrained(n_model)
n_model = AutoModelForSeq2SeqLM.from_pretrained(n_model)


summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, device=-1)
note_taker = pipeline("text2text-generation", model=n_model, tokenizer=n_tokenizer, device=-1)

max_worker = multiprocessing.cpu_count()

def summarize_text(text):
    return summarizer(text, min_length=5, max_length=500, do_sample=False)

def chunk_text(text, max_tokens=500):
    """
    Split text into chunks that fit within the model's max token length.
    Using 900 instead of 1024 for safety margin.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        current_chunk.append(word)
        # Count tokens if we added this word
        tokens = summarizer_tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"].shape[1]
        if tokens >= max_tokens:
            # Commit chunk and reset
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_chunk(chunk: str):
    """Summarize a single text chunk into bullet points."""
    prompt = f"Summarize the following text into important bullet points, stick to the language that the text is written in:\n\n{chunk}"
    result = summarizer(prompt, max_new_tokens=150, min_length=10, do_sample=False)
    text = result[0]["summary_text"]

    # Ensure bullet formatting
    bullets = [f"• {line.strip()}" for line in text.split("\n") if line.strip()]
    return "\n".join(bullets)

def notes_chunk(chunk: str):
    """Notes-style summarization (very important points only)."""
    prompt = f"""
    From the following text, extract only the very important points and write them as concise notes.
    - Use bullet points
    - Focus on facts, decisions, and action items
    - Ignore filler content
    - Stick to the language that the text is written in

    Text:
    {chunk}
    """

    result = note_taker(prompt, max_new_tokens=150, min_length=10, do_sample=False)
    text = result[0]["generated_text"]   # <- not summary_text

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
    if ext == ".mp4" or ext == ".mkv" or ext == ".webm" or ext == ".avi" or ext == ".mov":
        audio_file = f"{os.path.splitext(temp_file)[0]}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        os.remove(temp_file)  # cleanup original
        return audio_file
    elif ext == ".mp3" or ext == ".wav" or ext == ".ogg" or ext == ".flac":
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

    #transcribe
    transcript_text = transcribe_file(audio_file)

    # return {"transcript": transcript_text, "time": time.time() - start}


    # try:
    #     # Step 1: Transcribe
    #     print("Starting transcription...")
    #     t1 = time.time()
    #     segments, info = model.transcribe(audio_file)
    #     transcript_text = " ".join([seg.text for seg in segments])
    #     t2 = time.time()
    #     print(f"Transcription done in {t2 - t1:.2f} sec")

        # Step 2: Chunking
    try:
        chunks = chunk_text(transcript_text)

        # Step 3: Parallel summarization
        print("Starting summarization...")
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            summaries = list(executor.map(summarize_chunk, chunks))
        final_summary = " ".join(summaries)

        # Step 4: Generate notes
        print("Generating notes...")
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            notes = list(executor.map(notes_chunk, chunks))
        final_notes = " ".join(notes)

        print(f"Done in {time.time() - start:.2f} sec")

    finally:
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

    return {
        "transcript": transcript_text,
        "summary": final_summary,
        "notes": final_notes
    }

    

