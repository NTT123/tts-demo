import pickle

dic = pickle.load(open("./wavegru.ckpt", "rb"))
del dic["optim_state_dict"]
pickle.dump(dic, open("./wavegru.ckpt", "wb"))
