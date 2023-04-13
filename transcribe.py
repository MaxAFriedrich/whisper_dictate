import whisper


class Transcribe:
    def __init__(self, model) -> None:
        global MODEL
        print(model)
        MODEL = whisper.load_model(model)

    def run(self, frame_number: int, audio, host: str = "", port: int = 0):
        global MODEL
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
            print("Not text")
            return None
        return text
