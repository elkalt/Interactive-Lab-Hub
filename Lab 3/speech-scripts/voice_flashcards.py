import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice

SAMPLE_RATE = 16000
LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VAD = LAB_DIR / "models" / "silero_vad.onnx"
DEFAULT_VOICE = LAB_DIR / "voices" / "en_US-lessac-medium.onnx"
DEFAULT_CARDS = [
    {"front": "bonjour", "back": "hello", "hint": "A greeting you might say in the morning."},
    {"front": "merci", "back": "thank you", "hint": "Say this to show gratitude."},
    {"front": "chat", "back": "cat", "hint": "A small furry pet that says meow."},
    {"front": "livre", "back": "book", "hint": "You read this."},
    {"front": "rouge", "back": "red", "hint": "The color of a ripe strawberry."},
]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


class Flashcards:
    def __init__(self, voice_path: Path, vad_path: Path, model_name: str,
                 min_silence: float) -> None:
        self.voice = PiperVoice.load(str(voice_path))
        self.recognizer = WhisperModel(model_name, device="cpu", compute_type="int8")
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_path)
        config.silero_vad.min_silence_duration = min_silence
        config.silero_vad.min_speech_duration = 0.25
        config.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self.window = config.silero_vad.window_size

    def say(self, text: str) -> None:
        print(f"  speaking: {text}", flush=True)
        for chunk in self.voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            sd.play(audio, samplerate=chunk.sample_rate)
            sd.wait()

    def listen(self) -> str:
        print("  listening…", flush=True)
        audio_buffer = np.empty(0, dtype=np.float32)
        samples_per_read = int(0.1 * SAMPLE_RATE)
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as stream:
            while True:
                chunk, _ = stream.read(samples_per_read)
                audio_buffer = np.concatenate([audio_buffer, chunk.reshape(-1)])
                while len(audio_buffer) >= self.window:
                    self.vad.accept_waveform(audio_buffer[:self.window])
                    audio_buffer = audio_buffer[self.window:]
                if not self.vad.empty():
                    utterance = np.asarray(self.vad.front.samples, dtype=np.float32)
                    self.vad.pop()
                    segments, _ = self.recognizer.transcribe(utterance, beam_size=1)
                    return " ".join(segment.text.strip() for segment in segments).strip()


def load_cards(path: Path | None) -> list[dict[str, str]]:
    cards = DEFAULT_CARDS
    if path:
        try:
            cards = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read card deck {path}: {exc}") from exc
    if not isinstance(cards, list) or not cards:
        raise ValueError("The card deck must be a non-empty JSON array.")
    for number, card in enumerate(cards, 1):
        if not isinstance(card, dict) or not all(isinstance(card.get(k), str) and card[k].strip()
                                                 for k in ("front", "back")):
            raise ValueError(f"Card {number} needs non-empty 'front' and 'back' strings.")
        card.setdefault("hint", "")
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards", type=Path, help="JSON deck (front/back and optional hint fields)")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size (default: tiny.en)")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD)
    parser.add_argument("--min-silence", type=float, default=0.7,
                        help="silence in seconds before an answer is transcribed (default: 0.7)")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the deck")
    args = parser.parse_args()

    for file_path, label in ((args.voice, "Piper voice"), (args.vad_model, "VAD model")):
        if not file_path.is_file():
            sys.exit(f"{label} not found at {file_path}. Run speech-scripts/setup.sh first.")
    try:
        cards = load_cards(args.cards)
    except ValueError as exc:
        sys.exit(str(exc))
    if args.shuffle:
        random.shuffle(cards)

    print("Loading speech models…", flush=True)
    app = Flashcards(args.voice, args.vad_model, args.model, args.min_silence)
    score = 0
    app.say("Let's practice. Answer each card, or say hint, repeat, skip, or stop.")

    for index, card in enumerate(cards, 1):
        app.say(f"Card {index} of {len(cards)}. What does {card['front']} mean?")
        while True:
            answer = app.listen()
            print(f"  heard: {answer or '(no speech recognized)'}", flush=True)
            command = normalize(answer)
            if command in {"stop", "quit", "exit", "end"}:
                app.say(f"Session ended. You got {score} of {index - 1} cards correct. Au revoir!")
                return
            if command in {"repeat", "again", "say it again"} or not command:
                app.say(f"I'll repeat it. What does {card['front']} mean?")
                continue
            if command in {"hint", "help"}:
                app.say(card.get("hint", "No hint for this card. Try your best."))
                continue
            if command in {"skip", "pass", "next"}:
                app.say(f"We'll skip that one. The answer is {card['back']}.")
                break

            expected = normalize(card["back"])
            if command == expected:
                score += 1
                app.say("That's right!")
            else:
                app.say(f"Not quite. The answer is {card['back']}.")
            break

    app.say(f"Deck complete. You got {score} of {len(cards)} correct. Great work!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
