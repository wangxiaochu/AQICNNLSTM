# import the necessary packages
import models
from sklearn.model_selection import train_test_split,KFold,cross_val_predict
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,r2_score
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import glob
import os
import math
import pandas as pd
from datetime import datetime
from keras.applications import VGG16, ResNet50, VGG19,MobileNet,InceptionV3
from generator_cnnlstm import DataGenerator
import cv2
from keras.preprocessing.image import ImageDataGenerator


'''只用图像数据'''
if __name__ == '__main__':
	# load the dataset
	print("[INFO] loading data...")
	imagepath='Data/images/' #图像数据路径
	Y_attri = np.load(datapath + "Data/Y_attri.npy")
	X_image = np.load(datapath + "Data/X_image_includet.npy", allow_pickle=True)
	k_split = 1 #交叉验证的第k折
	## 交叉验证分割数据集
	print("[INFO] splitting data ...")
	kf = KFold(n_splits=2, random_state=10, shuffle=True)
	n_split = 0
	for train_index, test_index in kf.split(X_image,Y_attri):
		n_split = n_split + 1
		if n_split == k_split:
			trainImagelist, testImagelist = X_image[train_index], X_image[test_index]
			trainAttr, testAttr=Y_attri[train_index], Y_attri[test_index]

	# scale our data to the range [0, 1] (will lead to better training and convergence)
	# normalization AQI->500,PM2.5->500,Pm10->2000
	trainY_aqi = trainAttr[:,0]/ 500
	testY_aqi = testAttr[:,0]
	testY1_aqi = testAttr[:,0]/ 500
	trainY_pm2 = trainAttr[:, 1] / 500
	testY_pm2= testAttr[:, 1]
	testY1_pm2 = testAttr[:, 1] / 500
	trainY_pm10 = trainAttr[:, 2] / 2000
	testY_pm10 = testAttr[:, 2]
	testY1_pm10 = testAttr[:, 2] / 2000


	# load the images and then scale the pixel intensities to the range [0, 1]
	print("[INFO] loading images...")
	trainImagesX = models.load_imagefiles(imagepath,trainImagelist)
	testImagesX = models.load_imagefiles(imagepath, testImagelist)

	# 用CNN-LSTM对图像数据进行学习5
	print("[INFO] loading model...")
	model = models.transfer_cnnlstm2_multioutput(model=VGG16, input_shape=(7, 224, 224, 3), out_dim=1, mode='avg', regress=True)#model可改为ResNet50
	epoch_size=50
	batch_size=8
	opt = Adam(lr=1e-5)

	model.compile(loss={'out_aqi': "mean_squared_error",
						'out_pm2': "mean_squared_error",
						'out_pm10': "mean_squared_error"},
				  optimizer=opt,
				  metrics=[metrics.mae])
	print(model.summary())
	# train the model
	print("[INFO] training model...")
	result_path='result/vgglstm-t7/'+str(k_split)+'/'
	checkpath=result_path+'checkpoints/'
	TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
	if not os.path.exists(checkpath):
		os.makedirs(checkpath)

	callbacks_list=[ModelCheckpoint(filepath=checkpath + "model_{epoch}.h5",
									monitor='val_loss',verbose=1, save_best_only=True,
									save_weights_only=True,mode='auto', period=1),
					TensorBoard(log_dir=result_path+'logs/'+TIMESTAMP, histogram_freq=0,
								write_graph=True,write_images=True,write_grads=True)]


	training_generator = DataGenerator(images=trainImagesX, labels=[trainY_aqi,trainY_pm2,trainY_pm10], batch_size=batch_size, random_state=10)
	testing_generator = DataGenerator(images=testImagesX, labels=[testY1_aqi,testY1_pm2,testY1_pm10], batch_size=batch_size, shuffle=False)

	model.fit(x=training_generator,
			  validation_data=testing_generator,
			  epochs=epoch_size,
			  callbacks=callbacks_list)
	# save the trained model
	modelpath = result_path+'models/'
	if not os.path.exists(modelpath):
		os.makedirs(modelpath)
	modelfile = modelpath + 'mymodel.h5'
	model.save(modelfile)


	# make predictions on the testing data
	# AQI prediction
	print("[INFO] predicting...")
	preds = model.predict(x=testing_generator)
	preds=np.squeeze(np.array(preds))
	print(preds)
	predY_aqi = np.round(preds[0]*500,1)
	# finally, show some statistics on our model
	print("[INFO] avg. AQI: {}, std AQI: {}".format(testY_aqi.mean(),testY_aqi.std()))
	# compute the evaluate metrics between the *predicted* and the *actual*
	# print(len(testY),len(predY))
	r2_aqi=r2_score(testY_aqi,predY_aqi)
	ev_aqi=explained_variance_score(testY_aqi,predY_aqi)
	rmse_aqi=math.sqrt(mean_squared_error(testY_aqi,predY_aqi))
	mae_aqi=mean_absolute_error(testY_aqi,predY_aqi)
	print("[INFO] r2: {:.2f}, ev: {:.2f}, rmse: {:.2f}, mae: {:.2f}".format(r2_aqi, ev_aqi, rmse_aqi, mae_aqi))

	# PM2.5 prediction
	predY_pm2 = np.round(preds[1]*500,1)
	# finally, show some statistics on our model
	print("[INFO] avg. pm2.5: {}, std pm2.5: {}".format(testY_pm2.mean(),testY_pm2.std()))
	# compute the evaluate metrics between the *predicted* and the *actual*
	# print(len(testY),len(predY))
	r2_pm2=r2_score(testY_pm2,predY_pm2)
	ev_pm2=explained_variance_score(testY_pm2,predY_pm2)
	rmse_pm2=math.sqrt(mean_squared_error(testY_pm2,predY_pm2))
	mae_pm2=mean_absolute_error(testY_pm2,predY_pm2)
	print("[INFO] r2: {:.2f}, ev: {:.2f}, rmse: {:.2f}, mae: {:.2f}".format(r2_pm2, ev_pm2, rmse_pm2, mae_pm2))

	# PM10 prediction
	predY_pm10 = np.round(preds[2]*2000, 1)
	# finally, show some statistics on our model
	print("[INFO] avg. pm10: {}, std pm10: {}".format(testY_pm10.mean(), testY_pm10.std()))
	# compute the evaluate metrics between the *predicted* and the *actual*
	# print(len(testY),len(predY))
	r2_pm10 = r2_score(testY_pm10, predY_pm10)
	ev_pm10 = explained_variance_score(testY_pm10, predY_pm10)
	rmse_pm10 = math.sqrt(mean_squared_error(testY_pm10, predY_pm10))
	mae_pm10 = mean_absolute_error(testY_pm10, predY_pm10)
	print("[INFO] r2: {:.2f}, ev: {:.2f}, rmse: {:.2f}, mae: {:.2f}".format(r2_pm10, ev_pm10, rmse_pm10, mae_pm10))

	# save the predicted results
	PredTable=[]
	for ftest in range(len(testY_aqi)):
		PredTable.append({'Image Name': (testImagelist[ftest][-1]),
						  'True AQI': (testY_aqi[ftest]),
						  'Predict AQI': predY_aqi[ftest],
						  'True PM2.5': (testY_pm2[ftest]),
						  'Predict PM2.5': predY_pm2[ftest],
						  'True PM10': (testY_pm10[ftest]),
						  'Predict PM10': predY_pm10[ftest]})
	pred_df = pd.DataFrame(PredTable)
	print(pred_df)
	excel_name_test = checkpath+'预测结果'+str(k_split)+'.xlsx'
	pred_df.to_excel(excel_name_test)




