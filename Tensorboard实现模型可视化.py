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
with tf.name_scope('input'):           # 增加命名空间，以方便进行可视化
    x = tf.placeholder(float, [None, n_input], name='input_x')
    y = tf.placeholder(float, [None, n_classes], name='input_y')

# 构建多层神经网络

# def multilayer_perceptron(x ,weights, biases):
#     # 第一层  激活函数 ReLu
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#
#     # 第二层 激活函数 ReLu
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#
#     # 输出层
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer

def add_layer(x, input_tensors, output_tensors, activation_function=None):
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            weight = tf.Variable(tf.random_normal([input_tensors, output_tensors]), name='w')
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.random_normal([output_tensors]), name='b')
        with tf.name_scope('Wx_plus_b'):
            formula = tf.add(tf.matmul(x, weight), biases)
        if activation_function is None:
            output = formula
        else:
            output = activation_function(formula)
        return output

layer_1 = add_layer(x, input_tensors=n_input, output_tensors=n_hidden_1, activation_function=tf.nn.relu)
layer_2 = add_layer(layer_1, input_tensors=n_hidden_1, output_tensors=n_hidden_2, activation_function=tf.nn.relu)
out_layer = add_layer(layer_2, input_tensors=n_hidden_2, output_tensors=n_classes, activation_function=None)

# 设定代价函数
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# 训练模型
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('tensorboard/', graph=sess.graph)

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



