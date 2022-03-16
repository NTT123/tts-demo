import os

## build wavegru-cpp
os.system("./bazelisk-linux-amd64 clean --expunge")
os.system("./bazelisk-linux-amd64 build wavegru_mod -c opt --copt=-march=native")


import gradio as gr

from inference import load_tacotron_model, load_wavegru_net, mel_to_wav, text_to_mel
from wavegru_cpp import extract_weight_mask, load_wavegru_cpp

alphabet, tacotron_net, tacotron_config = load_tacotron_model(
    "./alphabet.txt", "./tacotron.toml", "./pretrained_model_ljs_500k.ckpt"
)


wavegru_config, wavegru_net = load_wavegru_net("./wavegru.yaml", "./wavegru.ckpt")

wave_cpp_weight_mask = extract_weight_mask(wavegru_net)
wavecpp = load_wavegru_cpp(wave_cpp_weight_mask)


def speak(text):
    mel = text_to_mel(tacotron_net, text, alphabet, tacotron_config)
    print(mel.shape)
    y = mel_to_wav(wavegru_net, wavecpp, mel, wavegru_config)
    print(y.shape)
    return 24_000, y


title = "WaveGRU-TTS"
description = "WaveGRU text-to-speech demo."

gr.Interface(
    fn=speak,
    inputs="text",
    outputs="audio",
    title=title,
    description=description,
    theme="default",
    allow_screenshot=False,
    allow_flagging="never",
).launch(debug=False)
