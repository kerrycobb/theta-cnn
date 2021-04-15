import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend
import pickle 

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def cnn(xTrain, yTrain, xTest, yTest):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation="relu",
            input_shape=(xTrain[0].shape[0], xTrain[0].shape[1]))) 
    model.add(Conv1D(64, kernel_size=2, activation="relu"))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer="normal"))
    # print(model.summary())
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[rmse])
    history = model.fit(xTrain, yTrain, epochs=10, batch_size=32, 
            validation_data=(xTest, yTest))
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_rmse = history.history["rmse"]
    val_rmse = history.history["val_rmse"]
    predicted = model.predict(xTest)
    output = dict(train_loss=train_loss, val_loss=val_loss, 
            train_rmse=train_rmse, val_rmse=val_rmse, predicted=predicted)
    return output 

length=50000

simData = np.load(f"simulated-data-{length}.npz")

populationSizes = simData["populationSizeArray"]
variantData = simData["variantMatrixArray"]
invariantData = simData["invariantMatrixArray"]

total = populationSizes.shape[0]
train = int(total - (total * .2))
print(f"Found {total} datasets")
print(f"Training on {train} datasets")
print(f"Testing on {total - train} datasets")

trainPopulationSizes = populationSizes[:train]
testPopulationSizes = populationSizes[train:]
trainVariant = variantData[:train] 
testVariant = variantData[train:] 
trainInvariant = invariantData[:train] 
testInvariant = invariantData[train:] 

variantOutput = cnn(trainVariant, trainPopulationSizes, testVariant, testPopulationSizes)
invariantOutput = cnn(trainInvariant, trainPopulationSizes, testInvariant, testPopulationSizes)
output = dict(variantOutput=variantOutput, invariantOutput=invariantOutput,
        testPopulationSizes=testPopulationSizes)

pickle.dump(output, open(f"cnn-output-n5000-len{length}-sam20-real.p", "wb"))

print("Analysis complete")