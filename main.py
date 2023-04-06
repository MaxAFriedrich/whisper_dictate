import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from pyautogui import typewrite
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


def transcribe(frame_number: int, audio):
    # concatinate audio
    audio = whisper.pad_or_trim(audio)

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


def bulk_transcribe(start: int, end: int):
    global audio_buffer
    full_audio = np.array([], dtype=np.float32)
    for i in range(start, end + 1):
        full_audio = np.concatenate([full_audio, audio_buffer.get(i)])
    return transcribe(end, full_audio)


# np.concatenate([old_audio, audio])
def transcribe_main(frame_size: float):
    global STOP
    frame_counter = 0
    current_start = None
    current_end = None
    final_text = None
    print("Transcribing")
    while not STOP:
        audio = audio_buffer.get(frame_counter)
        if audio is None:
            sleep(frame_size / 3)
        else:
            text = transcribe(frame_counter, audio)
            if not (
                text == ""
                or text == None
                or all(c in ". " for c in text)
                or check_for_repeating_chars(text)
            ):
                if current_start == None:
                    output(frame_counter, text)
                    current_end = current_start = frame_counter
                    final_text = text
                else:
                    current_end = frame_counter
                    new_text = bulk_transcribe(current_start, current_end)
                    if not (new_text == None or new_text == ""):
                        text = new_text
                    output(frame_counter, text)
                    final_text = text
            else:
                if final_text != None:
                    typewrite(final_text)
                    final_text = None
                if current_start == None or current_end == None:
                    audio_buffer.pop(frame_counter)
                else:
                    for i in range(current_start, current_end + 1):
                        audio_buffer.pop(i)
                current_start = current_end = None
            frame_counter += 1
    print("Not transcribing")


def output(frame_no: int, text: str):
    print(f"{frame_no}: {text}")

def listener(block_size:float):
    global STOP
    global audio_buffer
    print("Listening...")
    try:
        frame_number = 0
        while not STOP:
            audio_buffer[frame_number] = record(block_size)
            frame_number += 1

    except KeyboardInterrupt:
        STOP = True

    print("Not listening")

def main(block_size: float = 3):
    threading.Thread(target=transcribe_main, args=[block_size]).start()
    threading.Thread(target=listener,args=[block_size]).start()



if __name__ == "__main__":
    main()
