import pickle
with open("Backend/Models/dt_hog_model.pkl", "rb") as f:
    model = pickle.load(f)
print(type(model))
