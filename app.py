import gradio as gr
import os


## build wavegru-cpp
os.system("go get github.com/bazelbuild/bazelisk")
os.system("bazelisk build wavegru_mod -c opt --copt=-march=native")

from inference import load_tacotron_model, load_wavegru_net, text_to_mel, mel_to_wav
from wavegru_cpp import load_wavegru_cpp, extract_weight_mask


alphabet, tacotron_net, tacotron_config = load_tacotron_model(
    "./alphabet.txt", "./tacotron.toml", "./pretrained_model_ljs_500k.ckpt"
)


wavegru_config, wavegru_net = load_wavegru_net(
    "./wavegru.yaml", "./wavegru_vocoder_tpu_gta_preemphasis_pruning_v7_0040000.ckpt"
)

wave_cpp_weight_mask = extract_weight_mask(wavegru_net)
wavecpp = load_wavegru_cpp(wave_cpp_weight_mask)


def speak(text):
    mel = text_to_mel(tacotron_net, text, alphabet, tacotron_config)
    y = mel_to_wav(wavegru_net, wavecpp, mel, wavegru_config)
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
