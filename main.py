import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

X = pd.read_csv('household_consumption.csv')

data = pd.read_csv('household_consumption.csv', usecols=[1])
plt.plot(data)

# Convert pandas dataframe to numpy array
data = data.values
data = data.astype('float32')  # COnvert values to float

# --------------------------- Kmeans algorithm------------------------------------
# from sklearn.cluster import KMeans
# KMN= KMeans(n_clusters=3).fit(X1)
# #df=dataset.loc[:,[0,1]]

# #df=dataset.loc[:,['date','power']]
# #x=list(df['date'])s
# #y=list(df['power'])
# #KMN.fit(df)
# #predict labels or targets
# labels= KMN.predict(X1)
# #finding the centres defined by the model
# ctn=KMN.cluster_centers_

# # plt.scatter(X1,X[:,1],c=labels)
# # plt.scatter(ctn[:,0],ctn[:,1], marker= 'o', color='red', s=14)
# # plt.show()

# #finding the inertia of the mdoel
# KMN.inertia_
# k_inertia=[]

# for i in range(1,10):
#     KMN=KMeans(n_clusters=i, random_state=44)
#     KMN.fit(X1)
#     #appending the new data to the KMN.inertia_
#     k_inertia.append(KMN.inertia_)

# plt.plot(range(1,10),k_inertia, color='green', marker='o')
# plt.xlabel('number of k')
# plt.ylabel('inertia')
# plt.show()

# ------------------------------------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))  # Also try QuantileTransformer
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]


def ff(data, seq_size):
    x = []
    y = []

    for i in range(len(data) - seq_size - 1):
        #  w=X1[i:(i+seq_size,0)
        x.append(data[i:(i + seq_size), 0])
        y.append(data[i + seq_size, 0])

    return np.asarray(x), np.asarray(y)


seq_size = 5

trainX, trainY = ff(train, seq_size)
testX, testY = ff(test, seq_size)

# Compare trainX and dataset. You can see that X= values at t, t+1 and t+2
# whereas Y is the value that follows, t+3 (since our sequence size is 3)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('Single LSTM with hidden Dense...')
model = Sequential()
model.add(LSTM(46, input_shape=(1, 5)))
# model.add(Dense(32))
model.add(Dense(1))
model.compile()
model.compile(loss='mean_squared_error', optimizer='adam')
# monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20,
#                        verbose=1, mode='auto', restore_best_weights=True)
model.summary()
print('Train...')

model.fit()
model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=2, epochs=100)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainpredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testpredict = scaler.inverse_transform(trainPredict)
testY = scaler.inverse_transform([testY])
