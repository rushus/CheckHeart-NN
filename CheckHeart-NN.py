import pandas as pd
import keras
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

train_df=pd.read_csv('train_df.csv',header=None)
test_df=pd.read_csv('test_df.csv',header=None)

x = train_df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train_df = pd.DataFrame(x_scaled)

df_1=train_df[train_df[187]==1].sample(n=80,random_state=6)
df_2=train_df[train_df[187]==2].sample(n=80,random_state=7)
df_3=train_df[train_df[187]==3].sample(n=80,random_state=8)
df_4=train_df[train_df[187]==4].sample(n=80,random_state=9)
df_0=(train_df[train_df[187]==0]).sample(n=80,random_state=4)
train_df=pd.concat([df_0,df_1,df_2,df_3,df_4])

df_1=test_df[test_df[187]==1].sample(n=80,random_state=6)
df_2=test_df[test_df[187]==2].sample(n=80,random_state=7)
df_3=test_df[test_df[187]==3].sample(n=80,random_state=8)
df_4=test_df[test_df[187]==4].sample(n=80,random_state=9)
df_0=(test_df[test_df[187]==0]).sample(n=80,random_state=4)
test_df=pd.concat([df_0,df_1,df_2,df_3,df_4])

target_train=train_df[187]
target_test=test_df[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=train_df.iloc[:,:186].values
X_test=test_df.iloc[:,:186].values
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


def cnn(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], 1)
    inputs = Input(shape=(input_shape), name='inputs')

    x = Conv1D(64, (6), activation='relu', input_shape=input_shape)(inputs)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(3), strides=(2), padding="same")(x)

    x = Conv1D(64, (4), activation='relu', input_shape=input_shape, padding="same")(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(3), strides=(2), padding="same")(x)

    x = Dropout(0.2, name='dropout_one')(x)

    x = Conv1D(64, (4), activation='relu', input_shape=input_shape, padding="same")(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(3), strides=(2), padding="same")(x)

    x = Conv1D(64, (4), activation='relu', input_shape=input_shape, padding="same")(x)

    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=(3), strides=(2), padding="same")(x)

    x = Dropout(0.2, name='dropout_two')(x)

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(5, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=100),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, y_train, epochs=60, callbacks=callbacks, batch_size=4,
                        validation_data=(X_test, y_test))  # ep=40 bs=32
    model.load_weights('best_model.h5')
    return (model, history)


def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(history)

model, history = cnn(X_train,y_train,X_test,y_test)

evaluate_model(history,X_test,y_test,model)
y_pred=model.predict(X_test)