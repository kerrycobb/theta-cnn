import pickle 
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

output = pickle.load(open("cnn-output-n5000-len50000-sam20.p", "rb"))

variantOutput = output["variantOutput"]
invariantOutput = output["invariantOutput"]
populationSizes = output["testPopulationSizes"]

plt.plot(variantOutput["train_loss"])
plt.plot(variantOutput["val_loss"])
plt.title("Training MSE")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.savefig("plots/variant-model-loss.png")
plt.clf()

plt.plot(variantOutput["train_rmse"])
plt.plot(variantOutput["val_rmse"])
plt.title("Training RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.savefig("plots/variant-model-rmse.png")
plt.clf()

plt.plot(invariantOutput["train_loss"])
plt.plot(invariantOutput["val_loss"])
plt.title("Training MSE")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.savefig("plots/invariant-model-loss.png")
plt.clf()

plt.plot(invariantOutput["train_rmse"])
plt.plot(invariantOutput["val_rmse"])
plt.title("Training RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.savefig("plots/invariant-model-rmse.png")
plt.clf()

plt.plot(variantOutput["train_rmse"])
plt.plot(variantOutput["val_rmse"])
plt.plot(invariantOutput["train_rmse"])
plt.plot(invariantOutput["val_rmse"])
plt.title("Training RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend(["Variant Training", "Variant Validation", "Invariant Training", 
        "Invariant Validation"], loc="upper right")
plt.savefig("plots/both-model-rmse.png")
plt.clf()

plt.plot(variantOutput["train_loss"])
plt.plot(variantOutput["val_loss"])
plt.plot(invariantOutput["train_loss"])
plt.plot(invariantOutput["val_loss"])
plt.title("Training MSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend(["Variant Sites Training", "Variant Sites Validation", "All Sites Training", 
       "All Sites Validation"], loc="upper right")
plt.savefig("plots/both-model-loss.png")
plt.clf()

plt.scatter(populationSizes, variantOutput["predicted"])
plt.title("Variant Sites Only Prediction")
plt.xlabel("True θ")
plt.ylabel("Estamted θ")
plt.savefig("plots/variant-estimates.png")
plt.clf()

plt.scatter(populationSizes, invariantOutput["predicted"])
plt.title("All Sites Prediction")
plt.xlabel("True θ")
plt.ylabel("Estamted θ")
plt.savefig("plots/invariant-estimates.png")
plt.clf()

variant_rmse = mean_squared_error(variantOutput["predicted"],    
        populationSizes, squared=False) 
invariant_rmse = mean_squared_error(invariantOutput["predicted"], 
        populationSizes, squared=False) 
print(f"Variant rmse: {variant_rmse}")
print(f"Invariant rmse: {invariant_rmse}")