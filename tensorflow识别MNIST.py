import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\PythonCodes\Qiuyouwei_DL_Course\data', one_hot=True)
# print(type(mnist))
# print(type(mnist.train.images))
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)

# 呈现数字图像 其中images[2] 表示第二张图
# sample = mnist.train.images[2].reshape(28, 28)
# plt.imshow(sample)
# plt.show()

# 进行神经网络模型的构建

# 设置网络参数
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 设置两层神经网络模型，每一层的神经元的个数为256
# 隐藏层的神经元的个数一般会在输入的数据的维度的 1/2 ~ 1/3
n_hidden_1 = 256
n_hidden_2 = 256
# 输入层
n_input = 784
n_classes = 10
n_samples = mnist.train.num_examples # 获得数据集的整体的大小

# 定义占位符来接收输入与输出
x = tf.placeholder(float, [None, n_input])
y = tf.placeholder(float, [None, n_classes])

# 构建多层神经网络
def multilayer_perceptron(x ,weights, biases):
    # 第一层  激活函数 ReLu
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # 第二层 激活函数 ReLu
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 设定权重与偏差
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# 构建模型
pred = multilayer_perceptron(x, weights, biases)

# 代价函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# Xsample, Ysample = mnist.train.next_batch(1)
# plt.imshow(Xsample.reshape(28, 28))
# print(Ysample)

# 开始会话
sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        optimize, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

        # 计算平均loss
        # avg_cost += c / total_batch
        avg_cost += c
    avg_cost = avg_cost / total_batch
    print("Epoch: {}  cost={}".format(epoch + 1, avg_cost))
print("Training Completed in {} Epochs".format(training_epochs))

# 评估模型
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # argmax返回最大值的索引 1表示为按行， 0表示按列
# print(correct_prediction[0])  #发现是bool型
correct_prediction = tf.cast(correct_prediction, float)  # 使用cast函数转换成float型
# print(correct_prediction[0])

accuracy = tf.reduce_mean(correct_prediction)
print("Accuracy: ", accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))