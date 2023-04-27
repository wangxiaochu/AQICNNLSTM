import numpy as np
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels=None, attributes=None, batch_size=32, shuffle=True, random_state=42, aug=False):
        'Initialization'
        self.files = images
        self.labels = labels
        self.attributes = attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate image data
        x,y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # x=[]
        # y=[]
        # Aug = ImageDataGenerator(
        #     # rotation_range=30,
        #     zoom_range=0.15,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     # shear_range=0.15,
        #     horizontal_flip=True,
        #     fill_mode="nearest")
        # for k in indexes:
        #     img=cv2.imread(self.files[k])
        #     imgs = img.reshape((1,) + img.shape)
        #     if self.labels is not None:
        #         label = np.array(self.labels[k])
        #     if self.aug:
        #         a=Aug.flow(imgs,  batch_size=1, save_to_dir='E:/WXC/AQICNN/result/2020/AQI/1/', save_format='jpeg')
        #         i=0
        #         for batch in a:
        #             # print(next(a).shape)label,.reshape(img.shape)
        #             x.append(np.array(next(a))/255.0)
        #             y.append(np.array(label))
        #             i+=1
        #             if i>8: #扩增8个数据之后停止
        #                 break
        imgs = []
        images_batch = [self.files[k] for k in indexes]
        # Generate image data
        for img_files in images_batch:
            subimgs=[]
            for img_file in img_files:
                img = cv2.imread(img_file)
                ###############
                # Augment image
                ###############
                # img = cv2.resize(img, (224, 224))
                size=img.shape
                subimgs.append(img)#/255.0
            imgs.append(subimgs)
        # imgs =np.array(imgs).reshape(len(imgs), 1, size[0], size[1], size[2]) #当使用LSTM时，需要改变shape
        # print(np.array(imgs).shape)
        x = np.array(imgs)
        # print(x.shape)

        # output: training data and validation data need y
        # but testing data does not need y
        # y = []
        # if self.labels is not None:
        #     for k in indexes:
        #         y_label = self.labels[k]
        #         arr = np.zeros(size)
        #         arr[:, :, :] = y_label
        #         y.append(arr)
        #     y = np.array(y)
        if self.labels is not None:
            y=[]
            for i in range(len(self.labels)):
                batch_y = [self.labels[i][k] for k in indexes]
                y.append(np.array(batch_y))
        else:
            y = None
        # Augment image
        if self.aug:
            Aug = ImageDataGenerator(
                # rotation_range=30,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                # shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")
            x_gen = Aug.flow(x, batch_size=self.batch_size, shuffle=False)
            x=np.array(next(x_gen))

        x=x/255.0
        # multi-input
        # if self.attributes is not None:
        #     attr = [self.attributes[k] for k in indexes]
        #     attr = np.array(attr)
        #     x = [x, attr]
        if self.attributes is not None:
            attr=[]
            for k in indexes:
                at=[]
                for a in self.attributes[k]:
                    at.append(a)
                attr.append(at)
            attr = np.array(attr)
            x = [x, attr]
        # x=x.reshape(-1,size[0],size[1],size[2]) #当对图像使用LSTM时使用
        return x, y

    # def __getitem__(self, index):
    #     # Generate indexes of the batch
    #     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    #
    #
    #
    #     images_batch = [self.files[k] for k in indexes]
    #
    #     # Generate image data
    #     x = self.__data_generation(images_batch)
    #     x = np.array(x)
    #
    #     # multi-input
    #     if self.attributes is not None:
    #         attr = [self.attributes[k] for k in indexes]
    #         attr = np.array(attr)
    #         x=[x,attr]
    #
    #     # output: training data and validation data need y
    #     # but testing data does not need y
    #     if self.labels is not None:
    #         y = [self.labels[k] for k in indexes]
    #         y = np.array(y)
    #         return x, y
    #     else:
    #         return x
    #     # if self.aug:
    #     #     # construct the training image generator for data augmentation
    #     #     augment = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
    #     #                                  height_shift_range=0.2, shear_range=0.15,
    #     #                                  horizontal_flip=True, fill_mode="nearest")
    #     #     (x, y) = next(augment.flow(x, y, batch_size=self.batch_size))


    # def __data_generation(self, imgfiles):
    #     imgs = []
    #
    #     for img_file in imgfiles:
    #
    #         img = cv2.imread(img_file)
    #
    #         ###############
    #         # Augment image
    #         ###############
    #         # img = cv2.resize(img, (256, 256))
    #         size=img.shape
    #         imgs.append(img/255.0)
    #     # imgs = np.array(imgs).reshape(len(imgs), 1, size[0], size[1], size[2])
    #     return imgs


