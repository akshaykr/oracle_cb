import keras.models
import keras.layers.core

class NNModel(object):
    def __init__(self, num_layers, d, epochs=10):
        self.num_epochs = epochs
        self.model = keras.models.Sequential()
        if num_layers > 1:
            for i in range(num_layers-1):
                self.model.add(keras.layers.core.Dense(output_dim=d, init='glorot_uniform', activation='tanh', input_dim=d))
        self.model.add(keras.layers.core.Dense(1, init='glorot_uniform', activation='linear', input_dim=d))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X,y, sample_weight=None):
        if sample_weight is None:
            self.model.fit(X,y,nb_epoch=self.num_epochs, verbose=0)
        else:
            self.model.fit(X,y,sample_weight=sample_weight,nb_epoch=self.num_epochs, verbose=0)

    def predict(self, X):
        return (self.model.predict(X).T[0])
