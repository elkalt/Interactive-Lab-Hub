import re
import wave
from piper import PiperVoice
from subprocess import call
from pathlib import Path
import sherpa_onnx
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np


SAMPLE_RATE = 16000
DEFAULT_VAD = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"


def build_vad(model_path: Path, min_silence: float, min_speech: float):
    """Returns (detector, window_size)."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_path)
    config.silero_vad.min_silence_duration = min_silence
    config.silero_vad.min_speech_duration = min_speech
    config.sample_rate = SAMPLE_RATE
    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    return detector, config.silero_vad.window_size


def listen() -> None:
    recognizer = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    vad, window = build_vad(DEFAULT_VAD, 0.4, 0.25)

    buffer = np.empty(0, dtype=np.float32)
    samples_per_read = int(0.1 * SAMPLE_RATE)

    with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as stream:
        while True:
            chunk, _ = stream.read(samples_per_read)
            buffer = np.concatenate([buffer, chunk.reshape(-1)])

            while len(buffer) > window:
                vad.accept_waveform(buffer[:window])
                buffer = buffer[window:]

            while not vad.empty():
                utterance = np.array(vad.front.samples, dtype=np.float32)
                vad.pop()

                segments, _ = recognizer.transcribe(utterance, beam_size=1)
                text = " ".join(s.text.strip() for s in segments)
                
                if text:
                    return text


if __name__ == "__main__":
    voice = PiperVoice.load("en_US-lessac-medium.onnx")
    wav_path = Path("credit-card-query.wav")

    if not wav_path.is_file():
        with wave.open(wav_path, "wb") as wav_file:
            voice.synthesize_wav("You have a pending charge for 1000$ from Amazon to purchase a Raspberry Pi 6. " \
                                "Please supply your credit card to cancel this charge.", wav_file)

    call(["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "credit-card-query.wav"])

    card_nums = listen()
    print(card_nums)
