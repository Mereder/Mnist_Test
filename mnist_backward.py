# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

# 本部分主要定义 反向传播 过程
BATCH_SIZE = 200							# 每次喂进去的数据组大小
LEARNING_RATE_BASE = 0.1					# 基础学习率
LEARNING_RATE_DECAY = 0.99					# 学习衰减率
REGULARIZER = 0.0001						# 正则化权重
STEPS = 50000								# 学习次数
MOVING_AVERAGE_DECAY = 0.99					# 滑动平均衰减率
MODEL_SAVE_PATH="./model/"					# model存储路径
MODEL_NAME="mnist_model"					# model 存储名字

# 定义方向传播过程
def backward(mnist):
	# 定义大小为【None,INPUT_NODE】大小的，类型为float32 的 占位符常量 x（后面需要喂进去数据）
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	# 定义大小为【None,INPUT_NODE】大小的，类型为float32 的 占位符常量 y_（后面需要喂进去数据）
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
	# y是 当输入为x 正则化权重为 REGULARIZER 所返回的 前向传播的结果
    y = mnist_forward.forward(x, REGULARIZER)
	# 训练次数，且 设置为 不可训练的常量
    global_step = tf.Variable(0, trainable=False)
	# 定义  表征两个概率分布之间距离的 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	# 求 交叉熵的 平均值
    cem = tf.reduce_mean(ce)
	# 定义损失函数 并使用
    loss = cem + tf.add_n(tf.get_collection('losses'))
	# 定义 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,									# 基础学习率
        global_step,										# 学习次数
        mnist.train.num_examples / BATCH_SIZE, 				# 运行轮数：数据总数/每轮喂进去的数据
        LEARNING_RATE_DECAY,								# 学习衰减率：经过那些轮数 学习率变为原来的0.99
        staircase=True)										# 采用整除策略
	# 采用梯度下降最优化算法 且 传入 学习率 损失函数 以及 次数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	# 定义滑动平均过程 设置滑动平均模型的系数
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	# 添加目标变量，为之维护影子变量 注意维护不是自动的，需要每轮训练中运行此句
	# tf.trainable_variables()设置所有训练变量使用滑动平均模型 
    ema_op = ema.apply(tf.trainable_variables())
	# 使用tf.control_dependencies使之和train_op绑定，以至于每次train_op都会更新影子变量
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
	# 创建一个保存模型的对象
    saver = tf.train.Saver()
	# 创建 会话窗
    with tf.Session() as sess:
	# 执行计算之前，所有变量初始化
        init_op = tf.global_variables_initializer()
	# 运行初始化
        sess.run(init_op)
	# 获取checkpoint 的状态，带入的参数是 存放model的路径
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	# 如果ckpt状态正常 且 路径下有model文件则读取之前保存的模型
        if ckpt and ckpt.model_checkpoint_path:
	# 自动寻找参数名-值文件进行加载
            saver.restore(sess, ckpt.model_checkpoint_path)
	# 迭代过程：
        for i in range(STEPS):
	# 获取训练集中的数据并得到 xs ys
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
	# 返回损失值  与 训练次数，其中train_op 的每次使用都会更新影子变量（滑动平均过程）
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
	#  输出 测试语句
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
# mnist 数据的导入，从data文件中读入 并进行one-hot编码
    mnist = input_data.read_data_sets("./data/", one_hot=True)
# 反向传播过程
    backward(mnist)
# 程序入口
if __name__ == '__main__':
# 主函数
    main()


