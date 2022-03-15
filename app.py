import gradio as gr

from inference import load_tacotron_model, load_wavegru_net, text_to_mel, mel_to_wav

alphabet, tacotron_net, tacotron_config = load_tacotron_model(
    "./alphabet.txt", "./tacotron.toml", "./pretrained_model_ljs_500k.ckpt"
)


wavegru_config, wavegru_net = load_wavegru_net(
    "./wavegru.yaml", "./wavegru_vocoder_tpu_gta_preemphasis_pruning_v7_0040000.ckpt"
)


def speak(text):
    mel = text_to_mel(tacotron_net, text, alphabet, tacotron_config)
    y = mel_to_wav(wavegru_net, mel, wavegru_config)
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
