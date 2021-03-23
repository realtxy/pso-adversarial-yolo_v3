import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

from My_prediction import *
from core import utils
import shutil
import random


def is_point_in_box(center_point, box):
	if center_point[0]>box[0] and center_point[1]>box[1] \
			and center_point[0]<box[2]  and center_point[1]<box[3]:
		return True
	else:
		return False


def main():
	global config

	OutDir = './data/HideSpecificTarget_result/1/'
	Img_path = "data/coco_test_data/2.jpg"

	attack_index = 0  # 攻击第几个框  from 0
	continuous_round = 20
	mark = 0

	if os.path.exists(OutDir):
		shutil.rmtree(OutDir)
	os.makedirs(OutDir)

	pNumber = 3
	MaxEpoch = 20000  # 迭代上限

	Phi1 = 0.4
	Phi2 = 1.2
	Phi3 = 4
	# Phi4 = 1/20
	# Phi5 = 1/70

	IMAGE_H = 416
	IMAGE_W = 416

	with tf.Session(config=config) as sess:
		classes = utils.read_coco_names('./data/coco.names')

		###########对原图进行剪切获取SImg###############
		yolo = YoloTest()
		source_img = Image.open(Img_path)
		TImg = np.array(source_img.resize((IMAGE_H, IMAGE_W)), dtype=np.float32)/255.

		yolo.predict_from_array(TImg)
		yolo.last_PIL_image = Image.fromarray((255. * TImg).astype('uint8')).convert('RGB')
		yolo.save_result(os.path.join(OutDir, 'yuanshitupian.jpg' ))

		original_boxes, scores, labels = yolo.predict_from_array(TImg)
		assert len(labels) > attack_index
		# attack first label
		#attack_index = random.randint(0, len(labels) - 1)
		attack_index = random.choice(range(len(labels)))
		print("attack random label: {} ".format(classes[labels[attack_index]]))
		attcked_label = labels[attack_index]
		x_min = original_boxes[attack_index][0]
		y_min = original_boxes[attack_index][1]
		x_max = original_boxes[attack_index][2]
		y_max = original_boxes[attack_index][3]
		center_point = ((x_min+x_max)/2, (y_max+y_min)/2)

		Top_One_SM = np.zeros(shape=(pNumber), dtype=float)
		pbestFitness = np.zeros(shape=(pNumber), dtype=float)
		gbest = np.zeros(shape=(IMAGE_H, IMAGE_W, 3), dtype=float)
		gbestFitness = -1e8
		xOldOld = np.repeat(np.expand_dims(TImg, axis=0), pNumber, axis=0)
		xOld = np.repeat(np.expand_dims(TImg, axis=0), pNumber, axis=0)
		pbest = np.zeros(shape=(pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)
		x = np.zeros(shape=(pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)
		L2D = np.zeros(shape=(pNumber), dtype=float)
		Top_One_SM_pbest = np.zeros(shape=(pNumber), dtype=float)
		L2D_pbest = np.zeros(shape=(pNumber), dtype=float)
		Top_One_SM_gbest = 0.0
		L2D_gbest = 0.0

		sign_gbest = 0  # 标记是否有新的gbest生成
		Top_One_SM_SUM_round = np.ones(shape=(continuous_round), dtype=float)  # 记录每一轮的Top_One_SM_SUM
		x_round = np.zeros(shape=(continuous_round, pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)  # 记录每一轮的x
		xOld_round = np.zeros(shape=(continuous_round, pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)  # 记录每一轮的xOld
		xOldOld_round = np.zeros(shape=(continuous_round, pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)  # 记录每一轮的xOldOld
		gbest_round = np.zeros(shape=(continuous_round, pNumber, IMAGE_H, IMAGE_W, 3), dtype=float)  # 记录每一轮的gbest


		for i in range(MaxEpoch):
			print("[Epoch {}]################################################################################################################".format(i))
			if i==0:
				# initialization
				print( "initialization")
				for l in range(pNumber):
					x[l] = TImg + 2 * np.random.randn(IMAGE_H, IMAGE_W, 3)
					x[l] = np.clip(x[l], 0.0, 1.0)
					L2D[l] = np.linalg.norm(TImg - x[l])


				xOld = np.copy(x)
				xOldOld = np.copy(xOld)
				x_round[i] = np.copy(x)
				xOld_round[i] = np.copy(xOld)
				xOldOld_round[i] = np.copy(xOldOld)

				##########计算 标签得分##############
				for j in range(pNumber):
					boxes, scores, labels = yolo.predict_from_array(x[j])
					Top_One_SM[j] = 50
					for m, label in enumerate(labels):
						if label == attcked_label:
							# 如果原框还在，依据得分惩罚它
							if is_point_in_box(center_point, boxes[m]):
								Top_One_SM[j] = 0
								break

				Top_One_SM_SUM = 0
				for b in range(pNumber):
					Top_One_SM_SUM = Top_One_SM_SUM + Top_One_SM[b]
				Top_One_SM_SUM_round[i] = Top_One_SM_SUM

				fitness = Top_One_SM - L2D

				pbestFitness = np.copy(fitness)
				pbest = np.copy(x)
				Top_One_SM_pbest = np.copy(Top_One_SM)
				L2D_pbest = np.copy(L2D)

				for m in range(pNumber):
					if pbestFitness[m] > gbestFitness:
						gbestFitness = pbestFitness[m]
						gbest = np.copy(pbest[m])
						Top_One_SM_gbest = Top_One_SM[m]
						L2D_gbest = L2D[m]
				gbest_round[i] = np.copy(gbest)
				yolo.predict_from_array(gbest)
				yolo.last_PIL_image = Image.fromarray((255. * gbest).astype('uint8')).convert('RGB')
				yolo.save_result(os.path.join(OutDir, 'aaaaagbest_query_%d_TopOneSm_%.4f_L2_%.1f.jpg' %(i, Top_One_SM_gbest, L2D_gbest)))
				print("l2distance:", L2D )
				print("Label Score", Top_One_SM)

			else:


				if mark == 0:
					Top_One_SM_SUM_round_Sum = 0
					# 求 Top_One_SM_SUM_round中第1到最后一个元素的和
					for d in range(1, continuous_round):
						Top_One_SM_SUM_round_Sum += Top_One_SM_SUM_round[d]

					if bool(1 - (Top_One_SM_SUM_round[0] != 0 and Top_One_SM_SUM_round_Sum == 0)):
						pv = Phi1 * ((xOld - xOldOld) / 2 + Phi2 * np.random.rand() * (pbest - x) + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / 20 + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 70)
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
						print('Now in the Rapid iteration###################################################################################')
						if i <= continuous_round - 1:
							x_round[i] = np.copy(x)
						else:
							x_round[:continuous_round - 1] = np.copy(x_round[1:])
							x_round[continuous_round - 1] = np.copy(x)
					else:
						x = x_round[0]
						xOld = xOld_round[0]
						xOldOld = xOldOld_round[0]
						gbest = gbest_round[0]
						mark = 1

				else:
					if i < 30:
						guanxing = Phi1 * (xOld - xOldOld) / 2
						pbest_x = Phi1 * Phi2 * np.random.rand() * (pbest - x)
						gbest_x = Phi1 * Phi3 * np.random.rand() * (gbest - x)
						SImg_x = Phi1 * (TImg - x) / 100
						rand_num_list = Phi1 * np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 100
						pv = guanxing + pbest_x + gbest_x + SImg_x + rand_num_list
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
						print('x:', x[0][0][0])
						print('guanxing:', guanxing[0][0][0])
						print('pbest_x:', pbest_x[0][0][0])
						print('gbest_x:', gbest_x[0][0][0])
						print('SImg_x:', SImg_x[0][0][0])
						print('rand_num_list:', rand_num_list[0][0][0])
						print('pv:', pv[0][0][0])
					elif i < 100:
						pv = Phi1 * ((xOld - xOldOld) / 2 + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / 200 + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 70)
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
					elif i < 600:
						pv = Phi1 * ((xOld - xOldOld) / 4 + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / (0.65 * i + 170) + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 70)
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
					elif i < 1000:
						pv = Phi1 * ((xOld - xOldOld) / 5 + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / 400 + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 100)  # 此处pv并没有加约束，v使用的是标准正态分布，可以更改
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
					elif i < 1500:
						pv = Phi1 * ((xOld - xOldOld) / 5 + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / 600 + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 200)  # 此处pv并没有加约束，v使用的是标准正态分布，可以更改
						x = x + pv
						x = np.clip(x, 0.0, 1.0)
					else:
						pv = Phi1 * ((xOld - xOldOld) / 5 + Phi3 * np.random.rand() * (gbest - x) + (TImg - x) / 800 + np.random.randn(pNumber, IMAGE_H, IMAGE_W, 3) / 300)  # 此处pv并没有加约束，v使用的是标准正态分布，可以更改
						x = x + pv
						x = np.clip(x, 0.0, 1.0)


				xOldOld = np.copy(xOld)
				xOld = np.copy(x)


				for j in range(pNumber):
					L2D[j] = np.linalg.norm(TImg - x[j])
					boxes, scores, labels = yolo.predict_from_array(x[j])
					Top_One_SM[j] = 50
					for m, label in enumerate(labels):
						if label == attcked_label:
							# 如果原框还在，依据得分惩罚它
							if is_point_in_box(center_point, boxes[m]):
								Top_One_SM[j] = 0
								break


				fitness = Top_One_SM - L2D

				if mark == 0:
					Top_One_SM_SUM = 0
					for b in range(pNumber):
						Top_One_SM_SUM = Top_One_SM_SUM + Top_One_SM[b]

					if i <= continuous_round - 1:
						xOld_round[i] = np.copy(xOld)
						xOldOld_round[i] = np.copy(xOldOld)
						Top_One_SM_SUM_round[i] = Top_One_SM_SUM
					else:
						xOld_round[:continuous_round - 1] = np.copy(xOld_round[1:])
						xOld_round[continuous_round - 1] = np.copy(xOld)

						xOldOld_round[:continuous_round - 1] = np.copy(xOldOld_round[1:])
						xOldOld_round[continuous_round - 1] = np.copy(xOldOld)

						Top_One_SM_SUM_round[:continuous_round - 1] = Top_One_SM_SUM_round[1:]
						Top_One_SM_SUM_round[continuous_round - 1] = Top_One_SM_SUM



				for e in range(pNumber):
					##########更新pbest和pbestFitness###############
					if fitness[e] > pbestFitness[e]:
						pbestFitness[e] = fitness[e]
						pbest[e] = np.copy(x[e])
						Top_One_SM_pbest[e] = Top_One_SM[e]
						L2D_pbest[e] = L2D[e]

				for e in range(pNumber):
					##########更新gbest和gbestFitness###############
					if pbestFitness[e] > gbestFitness:
						sign_gbest = 1
						gbestFitness = pbestFitness[e]
						gbest = np.copy(pbest[e])
						Top_One_SM_gbest = Top_One_SM_pbest[e]
						L2D_gbest = L2D_pbest[e]

				if mark == 0:
					if i <= continuous_round - 1:
						gbest_round[i] = np.copy(gbest)
					else:
						gbest_round[:continuous_round - 1] = np.copy(gbest_round[1:])
						gbest_round[continuous_round - 1] = np.copy(gbest)
				if sign_gbest == 1:
					yolo.predict_from_array(gbest)
					yolo.last_PIL_image = Image.fromarray((255. * gbest).astype('uint8')).convert('RGB')
					yolo.save_result(os.path.join(OutDir, 'aaaaagbest_query_%d_TopOneSm_%.4f_L2_%.1f.jpg' % (i, Top_One_SM_gbest, L2D_gbest)))
				sign_gbest = 0



				# print()
				print("l2distance:", L2D)
				print("Label Score:", Top_One_SM)


if __name__ == '__main__':
	main()