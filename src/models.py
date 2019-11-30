import keras.backend as K
import keras.layers as KL
from keras import Input, Model


class LeNet:
    """
    Before calling mc-dropout, use 'set_mc_dropout_rate'.
    
    From mc-dropout paper:
    
        To assess model classification confidence in a realistic example we test a convolutional neural network trained on
        the full MNIST dataset (LeCun & Cortes, 1998). We trained the LeNet convolutional neural network model (LeCun et al., 1998) 
        with dropout applied before the last fully
        connected inner-product layer (the usual way dropout is
        used in convnets). We used dropout probability of 0.5. We
        trained the model for 10^6
        iterations with the same learning
        rate policy as before with γ = 0.0001 and p = 0.75. 
    """
    def __init__(self, input_shape, num_classes):
        self.mc_dropout_rate = K.variable(value=0)  # dropout before the last fully connected layer
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        inp = Input(shape=self.input_shape)
        x = KL.Conv2D(filters=20, kernel_size=5, strides=1)(inp)
        x = KL.MaxPool2D(pool_size=2, strides=2)(x)
        x = KL.Conv2D(filters=50, kernel_size=5, strides=1)(x)
        x = KL.MaxPool2D(pool_size=2, strides=2)(x)
        x = KL.Flatten()(x)
        x = KL.Dense(500, activation='relu')(x)
        x = KL.Lambda(lambda x: K.dropout(x, level=self.mc_dropout_rate))(x)  # dropout before the last fully connected layer
        x = KL.Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=inp, outputs=x, name='lenet-mc-dropout')
    
    def set_mc_dropout_rate(self, new_rate):
        K.set_value(self.mc_dropout_rate, new_rate)
        
    def train(self, X_train, y_train, X_test, y_test,
              batch_size=32,
              epochs=2,
              optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['acc'],
              verbose=0):
        
        print(f"Training with mc_dropout_rate = {K.eval(self.mc_dropout_rate)}.\n")
        model = self.model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # train the network
        model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=verbose,
        )
        
    def save_model(self, name):
        self.model.save_weights(f'{name}.h5')
    
    def load_model(self, path):
        self.model.load_weights(path)

