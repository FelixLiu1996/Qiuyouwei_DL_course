import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# hello = tf.constant("hello world")
# s = tf.Session()
# print(type(hello))
# print(s.run(hello))


# x = tf.constant(100)
# # print(x)
# print(type(x))
# s = tf.Session()
# print(s.run(x))
# print(type(s.run(x)))

# 进行操作
# x = tf.constant(2)
# y = tf.constant(3)

# 操作建议使用with操作
# sess = tf.Session()
# print(sess.run(x + y))
# sess.close()

# with tf.Session() as sess:
#     print(sess.run(x + y))
#     print(sess.run(x / y))

# 占位符
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.placeholder(float)
w = tf.placeholder(float, [2, None])
# print(x.shape)
d = {x : 20, y: 30}
add = tf.add(x, y)
with tf.Session() as sess:
    # print(sess.run(add, feed_dict=d))
    result = sess.run(w, feed_dict={w: [[1,2,2,2,2,2,2,2,2], [2,3,4,5,6,7,8,9,0]]})
    print(result)