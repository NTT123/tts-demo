import pickle

dic = pickle.load(open("./tacotrons_ljs_24k_v1_0300000.ckpt", "rb"))
del dic["optim_state_dict"]
pickle.dump(dic, open("./tacotrons_ljs_24k_v1_0300000.ckpt", "wb"))
