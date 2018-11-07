# coding:utf-8

# 本部分主要是前向传播搭建网络
import tensorflow as tf
# 定义输入结点、输出结点、中间层结点的数量
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 初始化 参数 w
def get_weight(shape, regularizer):
	# 用 去掉过大偏离点的正太分布，标准差为0.1生成 w 参数
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	# 损失函数loss 正则化
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 初始化 偏置项 b 为 全0
def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b
# 定义前向传播的神经网络结构
def forward(x, regularizer):
	# 通过 get_weight 获取 第一层与第二层的初始化的参数w1，且进行了正则化
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
	# 通过 get_bias 获取初始化的偏置项 b1
    b1 = get_bias([LAYER1_NODE])
	# 定义前向传播的神经网络结构 y1层是由x 通过参数w1 和偏置项b1 得到
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	# 通过 get_weight 获取 第2层与输出层的初始化的参数w2，且进行了正则化
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
	# 通过 get_bias 获取初始化的偏置项 b2
    b2 = get_bias([OUTPUT_NODE])
	# 定义前向传播的神经网络结构 输出层是由y1 通过参数w2 和偏置项b2 得到
    y = tf.matmul(y1, w2) + b2
    return y
