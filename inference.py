import os

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import pax

from text import english_cleaners
from utils import (
    create_tacotron_model,
    load_tacotron_ckpt,
    load_tacotron_config,
    load_wavegru_ckpt,
    load_wavegru_config,
)
from wavegru import WaveGRU

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "./espeak/usr/lib/libespeak-ng.so.1.1.51"
from phonemizer.backend import EspeakBackend

backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)


def load_tacotron_model(alphabet_file, config_file, model_file):
    """load tacotron model to memory"""
    with open(alphabet_file, "r", encoding="utf-8") as f:
        alphabet = f.read().split("\n")

    config = load_tacotron_config(config_file)
    net = create_tacotron_model(config)
    _, net, _ = load_tacotron_ckpt(net, None, model_file)
    net = net.eval()
    net = jax.device_put(net)
    return alphabet, net, config


tacotron_inference_fn = pax.pure(lambda net, text: net.inference(text, max_len=2400))


def text_to_mel(net, text, alphabet, config):
    """convert text to mel spectrogram"""
    text = english_cleaners(text)
    text = backend.phonemize([text], strip=True)[0]
    text = text + config["END_CHARACTER"]
    text = text + config["PAD"] * (100 - (len(text) % 100))
    tokens = []
    for c in text:
        if c in alphabet:
            tokens.append(alphabet.index(c))
    tokens = jnp.array(tokens, dtype=jnp.int32)
    mel = tacotron_inference_fn(net, tokens[None])
    return mel


def load_wavegru_net(config_file, model_file):
    """load wavegru to memory"""
    config = load_wavegru_config(config_file)
    net = WaveGRU(
        mel_dim=config["mel_dim"],
        rnn_dim=config["rnn_dim"],
        upsample_factors=config["upsample_factors"],
        has_linear_output=True,
    )
    _, net, _ = load_wavegru_ckpt(net, None, model_file)
    net = net.eval()
    net = jax.device_put(net)
    return config, net


wavegru_inference = pax.pure(lambda net, mel: net.inference(mel, no_gru=True))


def mel_to_wav(net, netcpp, mel, config):
    """convert mel to wav"""
    if len(mel.shape) == 2:
        mel = mel[None]
    pad = config["num_pad_frames"] // 2 + 2
    mel = np.pad(mel, [(0, 0), (pad, pad), (0, 0)], mode="edge")
    ft = wavegru_inference(net, mel)
    ft = jax.device_get(ft[0])
    wav = netcpp.inference(ft, 1.0)
    wav = np.array(wav)
    wav = librosa.mu_expand(wav - 127, mu=255)
    wav = librosa.effects.deemphasis(wav, coef=0.86)
    wav = wav * 2.0
    wav = wav / max(1.0, np.max(np.abs(wav)))
    wav = wav * 2**15
    wav = np.clip(wav, a_min=-(2**15), a_max=(2**15) - 1)
    wav = wav.astype(np.int16)
    return wav
