from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation="relu",
        input_shape=(10000, 80))) 
model.add(Conv1D(64, kernel_size=2, activation="relu"))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu", kernel_initializer="normal"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu", kernel_initializer="normal"))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer="normal"))

plot_model(model, to_file="keras-model.png", rankdir="LR", show_shapes=False, 
        show_layer_names=False)