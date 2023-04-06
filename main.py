import threading

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf

import whisper
from whisper.audio import SAMPLE_RATE

MODEL = whisper.load_model("base.en")
STOP = False


def write_audio_file(array, filename):
    # Used for debugging
    # Convert the array to 16-bit integer format
    array_int = np.int16(array * 32767)

    # Write the array to a WAV
    sf.write(filename, array_int, SAMPLE_RATE)


def record(duration: float, sr: int = SAMPLE_RATE):
    # Returns: A NumPy array containing the audio waveform, in float32 dtype
    try:
        # Record audio from default microphone for specified duration
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()  # Wait for recording to complete
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Failed to record audio from default microphone: {e}"
        ) from e

    return audio.flatten()


def transcribe(frame_number: int, audio, old_audio):
    # concatinate audio
    audio = whisper.pad_or_trim(np.concatenate([old_audio, audio]))

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)

    # decode the audio
    # joined_memory = "".join(frame_memory)
    options = whisper.DecodingOptions(
        fp16=False,
        language="en",
        prompt=" ",
        temperature=0,
    )
    # run the transcription
    try:
        result = whisper.decode(MODEL, mel, options)
        text = result.text
    except RuntimeError:
        print(f"Failed to transcribe {frame_number}.")
        return None
    if type(text) != str:
        return None
    return text


audio_buffer = {}
from time import sleep


def check_for_repeating_chars(text):
    words = text.split()
    if len(words) == 0:
        return False
    first_word = words[0]
    for word in words[1:]:
        if word != first_word:
            return False  # Words are not all the same
    return True  # All words are the same

def splice_at_deviation(old_text, text):
    """
    This function takes in two strings `old_text` and `text`, finds the first character that differs in `text`,
    and splices `old_text` and `text` together at that first deviation.

    Args:
    old_text: string, the original text that we want to splice into.
    text: string, the new text that we want to splice into `old_text`.

    Returns:
    A string that represents the spliced `old_text` and `text` together at their first deviation.
    """
    for i in range(min(len(old_text), len(text))):
        if old_text[i] != text[i]:
            return old_text[:i] + text[i:]

    # If all characters in `old_text` and `text` are the same, return the whole `text`
    return text

def transcribe_main(frame_size: float):
    global STOP
    old_text = ""
    merged_text = ""
    old_audio = np.array([], dtype=np.float32)
    stop_counter = 0
    frame_counter = 0
    print("Transcribing")
    while not STOP:
        audio = audio_buffer.get(frame_counter)
        if audio is None:
            sleep(frame_size / 3)
        else:
            text = transcribe(frame_counter, audio, old_audio)
            if not (
                text == ""
                or text == None
                or all(c in ". " for c in text)
                or check_for_repeating_chars(text)
            ):
                old_text = text
                merged_text = splice_at_deviation(merged_text,text)
                output(frame_counter, merged_text)
                old_audio = audio
                stop_counter = 0
            else:
                merged_text = ""
                stop_counter += 1
                print(stop_counter)
                if stop_counter > 3:
                    STOP = True
            audio_buffer.pop(frame_counter)
            frame_counter += 1
    print("Not transcribing")


def output(frame_no: int, text: str):
    print(f"{frame_no}: {text}")


def shortcut_listener(shortcut: str):
    global STOP

    def stop():
        global STOP
        STOP = True

    keyboard.add_hotkey(shortcut, stop)
    keyboard.wait()


def main(block_size: float = 5, shortcut: str = "ctrl+z"):
    global STOP
    # threading.Thread(target=shortcut_listener, args=[shortcut]).start()
    threading.Thread(target=transcribe_main, args=[block_size]).start()
    print("Listening...")
    try:
        frame_number = 0
        while not STOP:
            audio_buffer[frame_number] = record(block_size)
            frame_number += 1

    except KeyboardInterrupt:
        STOP = True

    print("Not listening")


if __name__ == "__main__":
    main()
