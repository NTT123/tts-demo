import numpy as np
from wavegru_mod import WaveGRU


def extract_weight_mask(net):
    data = {}
    data["embed_weight"] = net.embed.weight
    data["gru_h_zrh_weight"] = net.rnn.h_zrh_fc.weight
    data["gru_h_zrh_mask"] = net.gru_pruner.h_zrh_fc_mask
    data["gru_h_zrh_bias"] = net.rnn.h_zrh_fc.bias

    data["o1_weight"] = net.o1.weight
    data["o1_mask"] = net.o1_pruner.mask
    data["o1_bias"] = net.o1.bias
    data["o2_weight"] = net.o2.weight
    data["o2_mask"] = net.o2_pruner.mask
    data["o2_bias"] = net.o2.bias
    return data


def load_wavegru_cpp(data, repeat_factor):
    """load wavegru weight to cpp object"""
    embed = data["embed_weight"]
    rnn_dim = data["gru_h_zrh_bias"].shape[0] // 3
    net = WaveGRU(rnn_dim, repeat_factor)
    net.load_embed(embed)

    m = np.ascontiguousarray(data["gru_h_zrh_weight"].T)
    mask = np.ascontiguousarray(data["gru_h_zrh_mask"].T)
    b = data["gru_h_zrh_bias"]

    o1 = np.ascontiguousarray(data["o1_weight"].T)
    masko1 = np.ascontiguousarray(data["o1_mask"].T)
    o1b = data["o1_bias"]

    o2 = np.ascontiguousarray(data["o2_weight"].T)
    masko2 = np.ascontiguousarray(data["o2_mask"].T)
    o2b = data["o2_bias"]

    net.load_weights(m, mask, b, o1, masko1, o1b, o2, masko2, o2b)

    return net
