#!/usr/bin/env python3

import pickle 
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_line(data, params, title, ylabel, legend, filename):
    plt.clf()
    for i in data:
        for j in params:
            plt.plot(i[j])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig(f"{filename}.png")

def plot_prediction(data, title, filename):
    plt.clf()
    plt.scatter(data["simulated"], data["predicted"])
    plt.title(title)
    plt.xlabel("True θ")
    plt.ylabel("Estamted θ")
    plt.tight_layout()
    plt.savefig(f"{filename}.png")

var = pickle.load(open("var-model.p", "rb"))
pos_var = pickle.load(open("pos-var-model.p", "rb"))
invar = pickle.load(open("invar-model.p", "rb"))
red_var = pickle.load(open("red-var-model.p", "rb"))
red_pos_var = pickle.load(open("red-pos-var-model.p", "rb"))
red_invar = pickle.load(open("red-invar-model.p", "rb"))

print(var.keys())

plot_line(
    [red_var, var, red_pos_var, pos_var, red_invar, invar], 
    ["train_loss"],
    "Training Loss",
    "Loss", 
    ["Subset Variable Sites", "Variable Sites", 
        "Subset Variable Sites + Positions", "Variable Sites + Positions", 
         "Subset Invariant Sites", "Invariant Sites"],
    "plots/all-loss"
)

plot_line(
    [red_var, var, red_pos_var, pos_var, red_invar, invar], 
    ["train_rmse"],
    "Training RMSE",
    "RMSE", 
    ["Subset Variable Sites", "Variable Sites", 
        "Subset Variable Sites + Positions", "Variable Sites + Positions", 
         "Subset Invariant Sites", "Invariant Sites"], 
    "plots/all-rmse"
)

plot_prediction(red_var, "Subset Variable Sites", "plots/red-var-prediction")
plot_prediction(red_pos_var, "Subset Variable Sites + Positions", "plots/red-pos-var-prediction")
plot_prediction(red_invar, "Subset Invariant Sites", "plots/red-invar-prediction")
plot_prediction(var, "Variable Sites", "plots/var-prediction")
plot_prediction(pos_var, "Variable Sites + Positions", "plots/pos-var-prediction")
plot_prediction(invar, "Invariant Sites", "plots/invar-prediction")

# data = [red_var, var, red_pos_var, pos_var, red_invar, invar], 
# labels = ["Subset Variable Sites", "Variable Sites", 
#         "Subset Variable Sites + Positions", "Variable Sites + Positions", 
#          "Subset Invariant Sites", "Invariant Sites"]
# times = []
# for i in data:
#     times.append(sum(i["times"]))

# plt.clf()
# plt.bar(labels, times)
# # plt.title(title)
# # plt.xlabel("True θ")
# plt.ylabel("Time")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("plots/all-times.png")


# plot_loss([pos_var], 
#     "Loss", 
#     ["Pos Var"],
#     "summary-plots/pos-var-loss")

# plot_loss([var], 
#     "Loss", 
#     ["Var"],
#     "summary-plots/var-loss")

# plot_loss([invar], 
#     "Loss", 
#     ["Invar"],
#     "summary-plots/invar-loss")


# print(pos_var["times"])
# print(var["times"])
# print(invar["times"])


# print(output["times"])

# plt.plot(output["train_loss"])
# plt.plot(output["val_loss"])
# plt.title("Positions Training MSE")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Training", "Validation"], loc="upper right")
# plt.savefig("summary-plots/pos-var-mse.png")
# plt.clf()

# # plt.plot(output["train_rmse"])
# # plt.plot(output["val_rmse"])
# # plt.title("Positions Training RMSE")
# # plt.ylabel("RMSE")
# # plt.xlabel("Epoch")
# # plt.legend(["Training", "Validation"], loc="upper right")
# # plt.savefig("summary-plots/pos-var-rmse.png")
# # plt.clf()

# plt.scatter(output["simulated"], output["predicted"])
# plt.title("Positions Sites Only Prediction")
# plt.xlabel("True θ")
# plt.ylabel("Estamted θ")
# plt.savefig("summary-plots/pos-var-estimates.png")
# plt.clf()


# rmse = mean_squared_error(output["predicted"],    
#         output["simulated"], squared=False) 

# print(f"Position RMSE: {rmse}")

# ################################################################################

# output = pickle.load(open("var-model.p", "rb"))

# print(output["times"])

# plt.plot(output["train_loss"])
# plt.plot(output["val_loss"])
# plt.title("Variant Training MSE")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Training", "Validation"], loc="upper right")
# plt.savefig("summary-plots/var-mse.png")
# plt.clf()

# # plt.plot(output["train_rmse"])
# # plt.plot(output["val_rmse"])
# # plt.title("Variant Training RMSE")
# # plt.ylabel("RMSE")
# # plt.xlabel("Epoch")
# # plt.legend(["Training", "Validation"], loc="upper right")
# # plt.savefig("summary-plots/var-rmse.png")
# # plt.clf()

# plt.scatter(output["simulated"], output["predicted"])
# plt.title("Variant Sites Only Prediction")
# plt.xlabel("True θ")
# plt.ylabel("Estamted θ")
# plt.savefig("summary-plots/var-estimates.png")
# plt.clf()


# rmse = mean_squared_error(output["predicted"],    
#         output["simulated"], squared=False) 

# print(f"Variant RMSE: {rmse}")


# ################################################################################

# output = pickle.load(open("invar-model.p", "rb"))

# print(output["times"])

# plt.plot(output["train_loss"])
# plt.plot(output["val_loss"])
# plt.title("Invariant Training MSE")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Training", "Validation"], loc="upper right")
# plt.savefig("summary-plots/invar-mse.png")
# plt.clf()

# # plt.plot(output["train_rmse"])
# # plt.plot(output["val_rmse"])
# # plt.title("Invariant Training RMSE")
# # plt.ylabel("RMSE")
# # plt.xlabel("Epoch")
# # plt.legend(["Training", "Validation"], loc="upper right")
# # plt.savefig("summary-plots/invar-rmse.png")
# # plt.clf()

# plt.scatter(output["simulated"], output["predicted"])
# plt.title("Invariant Sites Prediction")
# plt.xlabel("True θ")
# plt.ylabel("Estamted θ")
# plt.savefig("summary-plots/invar-estimates.png")
# plt.clf()


# rmse = mean_squared_error(output["predicted"],    
#         output["simulated"], squared=False) 

# print(f"Invariant RMSE: {rmse}")