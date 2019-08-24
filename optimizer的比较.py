import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
df = pd.read_csv('D:\PythonCodes\Qiuyouwei_DL_Course\data\customer_churn.csv', index_col=0, header=0)
# print(df.head())
# print(df.columns)
# 前三列对于建模来说用处不大，所以先删除
# print(df.iloc[:,3:])
df = df.iloc[:, 3:]

# 进行数据预处理

# 先将类别型数据全都转换成数字
cat_var = ['international_plan', 'voice_mail_plan', 'churn']
for var in cat_var:
    df[var] = df[var].map(lambda x: 1 if x == 'yes' else 0)

# print(df.info())
y = df.iloc[:, -1] # 最后一列是label列
x = df.iloc[:, :-1] # 除了最后一列是数据列

# 区分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=123)  # random_state是seed
# print(x_train.shape)

# 标准化
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_train.shape)

# 对比不同optimizer
def trainProcess(optimizer):
    model = Sequential()
    model.add(Dense(units=8, input_dim=16, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=10, epochs=100, verbose=0, validation_data=(x_test, y_test))
    return history

history1 = trainProcess('sgd')
history2 = trainProcess('RMSprop')
history3 = trainProcess('Adagrad')
history4 = trainProcess('Adadelta')
history5 = trainProcess('Adam')

# 评估优化器
plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(history1.epoch, history1.history['acc'], label='SGD')
plt.plot(history2.epoch, history2.history['acc'], label='RMSProp')
plt.plot(history3.epoch, history3.history['acc'], label='Adagrad')
plt.plot(history4.epoch, history4.history['acc'], label='AdaDelta')
plt.plot(history5.epoch, history5.history['acc'], label='Adam')
plt.legend()
plt.show()