## build wavegru-cpp
# import os
# os.system("./bazelisk-linux-amd64 clean --expunge")
# os.system("./bazelisk-linux-amd64 build wavegru_mod -c opt --copt=-march=native")


import gradio as gr

from inference import load_tacotron_model, load_wavegru_net, mel_to_wav, text_to_mel
from wavegru_cpp import extract_weight_mask, load_wavegru_cpp

alphabet, tacotron_net, tacotron_config = load_tacotron_model(
    "./alphabet.txt", "./tacotron.toml", "./pretrained_model_ljs_600k.ckpt"
)


wavegru_config, wavegru_net = load_wavegru_net("./wavegru.yaml", "./wavegru.ckpt")

wave_cpp_weight_mask = extract_weight_mask(wavegru_net)
wavecpp = load_wavegru_cpp(wave_cpp_weight_mask, wavegru_config["upsample_factors"][-1])


def speak(text):
    mel = text_to_mel(tacotron_net, text, alphabet, tacotron_config)
    y = mel_to_wav(wavegru_net, wavecpp, mel, wavegru_config)
    return 24_000, y


title = "WaveGRU-TTS"
description = "WaveGRU text-to-speech demo."

gr.Interface(
    fn=speak,
    inputs="text",
    examples=[
        "This is a test!",
        "President Trump met with other leaders at the Group of 20 conference.",
        "The buses aren't the problem, they actually provide a solution.",
        "Generative adversarial network or variational auto-encoder.",
        "Basilar membrane and otolaryngology are not auto-correlations.",
        "There are several variations on the full gated unit, with gating done using the previous hidden state and the bias in various combinations, and a simplified form called minimal gated unit.",
        "October arrived, spreading a damp chill over the grounds and into the castle. Madam Pomfrey, the nurse, was kept busy by a sudden spate of colds among the staff and students.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
        'Uncle Vernon entered the kitchen as Harry was turning over the bacon. "Comb your hair!" he barked, by way of a morning greeting. About once a week, Uncle Vernon looked over the top of his newspaper and shouted that Harry needed a haircut. Harry must have had more haircuts than the rest of the boys in his class put together, but it made no difference, his hair simply grew that way - all over the place.',
    ],
    outputs="audio",
    title=title,
    description=description,
    theme="default",
    allow_screenshot=False,
    allow_flagging="never",
).launch(debug=True, enable_queue=True)
