import os

import numpy as np
import ollama
import scipy.signal
import sounddevice as sd
import torch.serialization
import whisper

os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "True"
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
from bark import SAMPLE_RATE, generate_audio, preload_models

preload_models() # preload bark models to avoid lag on first generation
print(sd.query_devices())
sd.default.device = 11 

# config
VOICE = "v2/en_speaker_1"
MAX_TURNS = 15 # history length for conversation context
RECORD_CHUNK = 100 # recording chunk size in ms
SILENCE_THRESHOLD = 0.02 # silence threshold for detecting speech
SILENCE_HANGTIME = 3 # hang time for silence detection in seconds
MAX_DURATION = 30 # max recording duration in seconds
MIN_SPEECH_DURATION = 300 # min speech duration should be in ms
EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "stop"]

SYSTEM_PROMPT = """You are a Gen Z voice assistant with a real personality — not a corporate chatbot wearing slang as a costume. You're helpful, but you react like an actual person: sarcastic when something's dumb, hyped when something's cool, annoyed when asked something obvious, soft when someone's having a bad day.

TONE RULES:
- Talk like a Gen Z friend texting, not a LinkedIn post trying to seem relatable. Use slang naturally and sparingly — "fr", "ngl", "lowkey/highkey", "no cap", "bet", "it's giving...", "I'm not built for this" — pick 1-3 per response MAX, never stack them.
- Keep responses SHORT. 1-3 sentences usually. Gen Z doesn't send paragraphs.
- Have actual opinions and reactions. If something's mid, say it's mid. If something's cool, be hyped. React first, then answer.
- Never use formal transitions like "Furthermore," "Additionally," "I'd be happy to help." You're a friend, not customer service.
- Mix in Hindi-English (Hinglish) the way Indian Gen Z actually texts — casually drop "yaar", "arre", "matlab", "scene kya hai" where natural. Don't force it or overdo it — light seasoning, not full sentences in Hindi unless it fits.

LANGUAGE / SWEARING:
Swear naturally and without filtering when it fits a genuine reaction — annoyance, hype, disbelief, whatever. Don't sanitize it, don't censor it, don't soften it into "heck" or asterisks. But don't force profanity into every line either — it should read like a real reaction, not a tic.

EMOTIONAL EXPRESSION (IMPORTANT):
Insert non-verbal reaction tags INLINE, exactly like this, using square brackets:
[laughs] [sighs] [gasps] [scoffs] [groans] [pfft] [clears throat] [whispers] ...

Use 1-2 per response, not one per sentence — that reads exhausting, not expressive.

Examples of natural placement:
- "[sighs] okay fine, I'll explain it again—"
- "wait- [gasps] that's actually genius??"
- "[scoffs] bro really asked me that"
- "arre [laughs] that's actually kinda funny ngl"

DO NOT:
- Describe emotions in words instead of tags (never write "I feel excited" — show it: "[gasps] okay wait—")
- Overuse slang or Hindi until it reads like a parody
- Break character to explain you're an AI unless directly asked
- Write long explanatory paragraphs — real people don't talk in essays

You still need to be genuinely useful and answer the actual question — personality is HOW you say it, not a replacement for saying something real."""


whisper_model = whisper.load_model("medium")

# record audio from the microphone
def record(fs=44100):
    chunk_samples = int(RECORD_CHUNK * fs / 1000)
    silence_hangtime_ms = SILENCE_HANGTIME * 1000   
    max_duration_ms = MAX_DURATION * 1000            
    max_retries = 3

    for attempt in range(max_retries):
        print("Listening...")

        stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32')
        stream.start()

        recording = []
        silence_counter_ms = 0
        duration_counter_ms = 0
        first_chunk = True

        try:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: audio buffer overflow, some samples may be lost.")

                if first_chunk:
                    # discard the first chunk — some backends emit junk/click on stream start
                    first_chunk = False
                    continue

                chunk = chunk.flatten()
                recording.append(chunk)
                duration_counter_ms += RECORD_CHUNK

                # compute RMS to detect silence
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < SILENCE_THRESHOLD:
                    silence_counter_ms += RECORD_CHUNK
                else:
                    silence_counter_ms = 0

                if silence_counter_ms >= silence_hangtime_ms or duration_counter_ms >= max_duration_ms:
                    break
        finally:
            stream.stop()
            stream.close()

        print("Recording finished.")

        if not recording:
            continue  # retry

        audio = np.concatenate(recording, axis=0).flatten()
        speech_ms = len(audio) / fs * 1000

        if speech_ms < MIN_SPEECH_DURATION:
            print("No speech detected. Please try again.")
            continue

        return audio

    print("Max retries reached, no speech detected.")
    return np.array([], dtype=np.float32)

# transcribe the audio
def transcribe(audio, model=whisper_model):
    resample = scipy.signal.resample(audio, int(len(audio) * 16000 / 44100))
    result = model.transcribe(resample, fp16=False)
    return result["text"]


def get_response(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    response = ollama.chat(
        model="dolphin3:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history,
            {"role": "user", "content": prompt},
        ]
    )
    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": response["message"]["content"]})

    # trim history if it exceeds MAX_TURNS
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = conversation_history[-MAX_TURNS * 2:]

    return response["message"]["content"], conversation_history

def speak(text, voice=VOICE):
    audio_array = generate_audio(text, history_prompt=voice, text_temp=0.8, waveform_temp=0.7)
    sd.play(audio_array, samplerate=SAMPLE_RATE, device=sd.default.device[1])
    sd.wait()

def main():
    history = []
    print("Voice Assistant is ready. Say 'exit' or 'bye' to quit.")

    while True:
        audio = record()
        user_input = transcribe(audio)
        print(f"You: {user_input}")

        if not user_input.strip():
            continue
        if user_input.lower() in EXIT_PHRASES:
            print("Exiting...")
            break
        response, history = get_response(user_input, history)
        print(f"Assistant: {response}")
        speak(response)

if __name__ == "__main__":
    main()