import whisper
from whisper.audio import SAMPLE_RATE
import sounddevice as sd
import threading
import numpy as np

MODEL = whisper.load_model("base.en")

def record(duration:float, sr: int = SAMPLE_RATE):
    try:
        # Record audio from default microphone for specified duration
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to complete
    except sd.PortAudioError as e:
        raise RuntimeError(f"Failed to record audio from default microphone: {e}") from e

    return audio.flatten()

# frame_memory = []
# previous=""
# old_audio = None
def transcribe(frame_number:int, audio):
    # global previous,old_audio
    # load audio and pad/trim it to fit 30 seconds
    # if old_audio == None:
    #     tmp_audio = audio
    # else:
    #     tmp_audio = np.concatenate((old_audio,audio),axis=0)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)

    # decode the audio
    # joined_memory = "".join(frame_memory)
    options = whisper.DecodingOptions(fp16=False,language="en")#,prefix=previous)
    result = whisper.decode(MODEL, mel, options)


    # add audio to memory
    # frame_memory.append(result.text)
    # if len(frame_memory) > 30:
    #     frame_memory.pop(0)

    # previous = result.text
    # old_audio=audio
    # print the recognized text
    output(frame_number,result.text)


def output( frame_no:int, text:str):
    print(f"{frame_no}: {text}")

def main(block_size:float= 5):
    print("Listening...")
    try:
        frame_number = 0
        while True:
            audio = record(block_size)

            threading.Thread(target=transcribe,args=(frame_number,audio)).start()
            frame_number+=1

    except KeyboardInterrupt:
        print("Stopped")

if __name__ == "__main__":
    main()
