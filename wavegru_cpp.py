import numpy as np
import sys

sys.path.append("./bazel-bin")
from wavegru_mod import WaveGRU

sys.path.pop()


def extract_weight_mask(net):
    data = {}
    data["embed_weight"] = net.embed.weight
    data["gru_xh_zr_weight"] = net.rnn.xh_zr_fc.weight
    data["gru_xh_zr_mask"] = net.gru_pruner.xh_zr_fc_mask
    data["gru_xh_zr_bias"] = net.rnn.xh_zr_fc.bias

    data["gru_xh_h_weight"] = net.rnn.xh_h_fc.weight
    data["gru_xh_h_mask"] = net.gru_pruner.xh_h_fc_mask
    data["gru_xh_h_bias"] = net.rnn.xh_h_fc.bias

    data["o1_weight"] = net.o1.weight
    data["o1_mask"] = net.o1_pruner.mask
    data["o1_bias"] = net.o1.bias
    data["o2_weight"] = net.o2.weight
    data["o2_mask"] = net.o2_pruner.mask
    data["o2_bias"] = net.o2.bias
    return data


def load_wavegru_cpp(data):
    embed = data["embed_weight"]
    embed_dim = embed.shape[1]
    rnn_dim = data["gru_xh_h_bias"].shape[0]
    input_dim = data["gru_xh_zr_weight"].shape[1] - rnn_dim
    net = WaveGRU(input_dim, embed_dim, rnn_dim)
    net.load_embed(embed)
    dim = embed_dim + input_dim + rnn_dim
    z, r = np.split(data["gru_xh_zr_weight"].T, 2, axis=0)
    h = data["gru_xh_h_weight"].T
    z = np.ascontiguousarray(z)
    r = np.ascontiguousarray(r)
    h = np.ascontiguousarray(h)

    b1, b2 = np.split(data["gru_xh_zr_bias"], 2)
    b3 = data["gru_xh_h_bias"]
    m1, m2, m3 = z, r, h

    mask_z, mask_r = np.split(data["gru_xh_zr_mask"].T, 2, axis=0)
    mask_h = data["gru_xh_h_mask"].T
    mask_z = np.ascontiguousarray(mask_z)
    mask_r = np.ascontiguousarray(mask_r)
    mask_h = np.ascontiguousarray(mask_h)

    mask1, mask2, mask3 = mask_z, mask_r, mask_h

    o1 = np.ascontiguousarray(data["o1_weight"].T)
    masko1 = np.ascontiguousarray(data["o1_mask"].T)
    o1b = data["o1_bias"]

    o2 = np.ascontiguousarray(data["o2_weight"].T)
    masko2 = np.ascontiguousarray(data["o2_mask"].T)
    o2b = data["o2_bias"]

    net.load_weights(
        m1,
        mask1,
        b1,
        m2,
        mask2,
        b2,
        m3,
        mask3,
        b3,
        o1,
        masko1,
        o1b,
        o2,
        masko2,
        o2b,
    )

    return net
