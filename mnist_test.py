#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
# 测试间隔数
TEST_INTERVAL_SECS = 5

# 本部分主要为测试函数
def test(mnist):
# 定义TensorFlow计算图
    with tf.Graph().as_default() as g:
	# 定义大小为【None,INPUT_NODE】大小的，类型为float32 的 占位符常量 x（后面需要喂进去数据）
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	# 定义大小为【None,INPUT_NODE】大小的，类型为float32 的 占位符常量 y_（后面需要喂进去数据）
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
	# y是 当输入为x 正则化权重为 REGULARIZER 所返回的 前向传播的结果
        y = mnist_forward.forward(x, None)
# 定义滑动平均过程 设置滑动平均模型的系数
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
# 在加载模型的时候将影子变量直接映射到变量的本身，在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量。
        ema_restore = ema.variables_to_restore()
# 实例化还原滑动平均值的 Saver
        saver = tf.train.Saver(ema_restore)
# 定义准确率的计算方法
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 对所有准确率进行求平均，作为最终的 准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
		# 创建会话
            with tf.Session() as sess:
			# 加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			# 判断是否存在 checkpoint的模型
                if ckpt and ckpt.model_checkpoint_path:
					# 从checkpoint中 恢复会话
                    saver.restore(sess, ckpt.model_checkpoint_path)
					# 恢复轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					# 计算准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
					# 输出准确率
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
				# 如果checkpoint 存在问题 则打印错误信息
                    print('No checkpoint file found')
                    return
			# 一次结束后 进行休息5 ms 然后再进入下一轮
            time.sleep(TEST_INTERVAL_SECS)
# 主函数
def main():
# 从mnist中读取数据集
    mnist = input_data.read_data_sets("./data/", one_hot=True)
# 对数据集进行测试
    test(mnist)
# 程序入口
if __name__ == '__main__':
# 主函数
    main()
