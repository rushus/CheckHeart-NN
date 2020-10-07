import pandas as pd
import keras
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

train_df=pd.read_csv('',header=None)
test_df=pd.read_csv('',header=None)

target_train=train_df[:]
target_test=test_df[:]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=train_df.iloc[:].values
X_test=test_df.iloc[:].values
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)

def cnn(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], 1)
    inputs = Input(shape=(input_shape), name='inputs')

    x = Conv1D(64, (6), activation='relu', input_shape=input_shape)(inputs)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(3), strides=(2), padding="same")(x)

    x = Conv1D(64, (3), activation='relu', input_shape=input_shape)(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(2), strides=(2), padding="same")(x)

    x = Dropout(0.2, name='dropout_one')(x)

    x = Conv1D(64, (3), activation='relu', input_shape=input_shape)(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(2), strides=(2), padding="same")(x)

    x = Conv1D(64, (3), activation='relu', input_shape=input_shape)(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(2), strides=(2), padding="same")(x)

    x = Dropout(0.2, name='dropout_two')(x)

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(5, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, y_train, epochs=40, callbacks=callbacks, batch_size=32,
                        validation_data=(X_test, y_test))
    model.load_weights('best_model.h5')
    return (model, history)


def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(history)

model, history = cnn(X_train,y_train,X_test,y_test)

evaluate_model(history,X_test,y_test,model)
y_pred=model.predict(X_test)