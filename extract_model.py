import pickle

dic = pickle.load(open("./wavegru_vocoder_1024_v3_1340000.ckpt", "rb"))
del dic["optim_state_dict"]
pickle.dump(dic, open("./wavegru_vocoder_1024_v3_1340000.ckpt", "wb"))
