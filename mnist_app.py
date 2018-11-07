# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward
import sys 

#  重新加载 model
def restore_model(testPicArr):
# 定义计算图
	with tf.Graph().as_default() as tg:
	# 定义大小为【None,INPUT_NODE】大小的，类型为float32 的 占位符常量 x（后面需要喂进去数据）
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	# 获取前向传播的过程的结果
		y = mnist_forward.forward(x, None)
	# 获取预测的值
		preValue = tf.argmax(y, 1)
		# 定义变量的滑动平均
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		# 预备可 存储的变量
		variables_to_restore = variable_averages.variables_to_restore()
		# 存储模型中 填入滑动平均的变量值
		saver = tf.train.Saver(variables_to_restore)
		# 定义新的会话模型
		with tf.Session() as sess:
		# 加载ckpt模型
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
		# 判断是否存在 checkpoint的模型
			if ckpt and ckpt.model_checkpoint_path:
			# 若存在则 从checkpoint中 恢复会话
				saver.restore(sess, ckpt.model_checkpoint_path)
			# 获取预测的结果
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
			# 若不存在 则打印错误信息
				print("No checkoint file found")
				return -1
# 对图片进行预处理的函数
def pre_pic(picName):
# 打开图片
	img = Image.open(picName)
# 对图片大小进行重新定义
	reIm = img.resize((28,28), Image.ANTIALIAS)
# 定义图片的数组
	im_arr = np.array(reIm.convert('L'))
# 定义最小的阈值
	threshold = 50
# 对图片进行处理
	for i in range(28):
		for j in range(28):
		# 对图片进行二值处理
			im_arr[i][j] = 255 - im_arr[i][j]
		# 如果小于某个阈值 则置为0
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
		# 否则 置为 255
			else: im_arr[i][j] = 255
			
# 将28*28的矩阵转化为 1*784的向量
	nm_arr = im_arr.reshape([1,784])
# 重新定义该向量的数据类型
	nm_arr = nm_arr.astype(np.float32)
# 进行归一化处理，（之前不是0 就是 255  都* 1/255 之后 变为 0 或 1）
	img_ready = np.multiply(nm_arr, 1.0/255.0)
# 返回处理好的图像数据
	return img_ready
# 应用函数入口
def application():
# 获取测试的图片数量
	testNum = input("input the number of test pictures:")
	# 稍作改动的代码 可以从命令行直接输入图片
	# testNum = len(sys.argv)-1
	# testPic = sys.argv[1:]
# 对待测试图片进行测试
	for i in range(testNum):
	# 输入测试图片的路径
		testPic = raw_input("the path of test picture:")
	# 调用pre_pic函数 对图片进行预处理
		testPicArr = pre_pic(testPic)
	# 获得预测的值
		preValue = restore_model(testPicArr)
	# 输出预测值
		print "The prediction number is:", preValue
# 主函数定义
def main():
# 调用应用函数
	application()
# 程序入口
if __name__ == '__main__':
# 调用主函数
	main()