"""
Utility functions
"""
import pickle
from pathlib import Path

import pax
import toml
import yaml

from tacotron import Tacotron


def load_tacotron_config(config_file=Path("tacotron.toml")):
    """
    Load the project configurations
    """
    return toml.load(config_file)["tacotron"]


def load_tacotron_ckpt(net: pax.Module, optim: pax.Module, path):
    """
    load checkpoint from disk
    """
    with open(path, "rb") as f:
        dic = pickle.load(f)
    if net is not None:
        net = net.load_state_dict(dic["model_state_dict"])
    if optim is not None:
        optim = optim.load_state_dict(dic["optim_state_dict"])
    return dic["step"], net, optim


def create_tacotron_model(config):
    """
    return a random initialized Tacotron model
    """
    return Tacotron(
        mel_dim=config["MEL_DIM"],
        attn_bias=config["ATTN_BIAS"],
        rr=config["RR"],
        max_rr=config["MAX_RR"],
        mel_min=config["MEL_MIN"],
        sigmoid_noise=config["SIGMOID_NOISE"],
        pad_token=config["PAD_TOKEN"],
        prenet_dim=config["PRENET_DIM"],
        attn_hidden_dim=config["ATTN_HIDDEN_DIM"],
        attn_rnn_dim=config["ATTN_RNN_DIM"],
        rnn_dim=config["RNN_DIM"],
        postnet_dim=config["POSTNET_DIM"],
        text_dim=config["TEXT_DIM"],
    )


def load_wavegru_config(config_file):
    """
    Load project configurations
    """
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wavegru_ckpt(net, optim, ckpt_file):
    """
    load training checkpoint from file
    """
    with open(ckpt_file, "rb") as f:
        dic = pickle.load(f)

    if net is not None:
        net = net.load_state_dict(dic["net_state_dict"])
    if optim is not None:
        optim = optim.load_state_dict(dic["optim_state_dict"])
    return dic["step"], net, optim
