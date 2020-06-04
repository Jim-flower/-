import tensorflow as tf
import os               #导入模块
from    tensorflow import keras
from    tensorflow.keras import layers
from    tensorflow.keras.applications import resnet
from tensorflow.keras import layers,Sequential
import cv2
import numpy as np
from PIL import Image
from try_1 import *
class BasicBlock(layers.Layer):
    def __init__(self,filter_num,stride =1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=stride,padding="same",)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=stride,padding="same")
        self.bn2 = layers.BatchNormalization()
        if stride !=1:
            self.downsaple = Sequential()
            self.downsaple.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsaple = lambda x:x
    def call(self, inputs, traing =None):

        out = self.conv1(inputs)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        identity = self.downsaple(inputs)
        output =layers.add([out,identity])

        output = tf.nn.relu(output)
        return output
class layerone(layers.Layer):

    def __init__(self,filter_num,stride =1):
        super(layerone, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=stride,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=stride,padding="same")
        self.bn2 = layers.BatchNormalization()
        if stride !=1:
            self.downsaple = Sequential()
            self.downsaple.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsaple = lambda x:x

    def call(self, inputs, traing =None):

        out = self.conv1(inputs)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        identity = self.downsaple(inputs)
        output =layers.add([out,identity])

        output = tf.nn.relu(output)
        return output

def first(filename, label):
    # print(type(filename))
    file1_name = filename+"/1.png"
    file2_name = filename + "/2.png"
    file3_name = filename + "/3.png"
    file4_name = filename + "/4.png"
    file5_name = filename + "/phase.png"
    image1 = convert_image(file1_name)
    image2 = convert_image(file2_name)
    image3 = convert_image(file3_name)
    image4 = convert_image(file4_name)
    image5 = convert_image(file5_name)
    image = tf.concat([image1,image2],2)
    image = tf.concat([image, image3], 2)
    image = tf.concat([image, image4], 2)
    # labe =  convert_image(label)
    label = image5
    return image, label

def second(filename):
    # print(type(filename))
    file1_name = filename+"/1.png"
    file2_name = filename + "/2.png"
    file3_name = filename + "/3.png"
    file4_name = filename + "/4.png"
    # file5_name = filename + "/5.png"
    image1 = convert_image(file1_name)
    image2 = convert_image(file2_name)
    image3 = convert_image(file3_name)
    image4 = convert_image(file4_name)
    # image5 = convert_image(file5_name)
    image = tf.concat([image1,image2],2)
    image = tf.concat([image, image3], 2)
    image = tf.concat([image, image4], 2)
    # labe =  convert_image(label)
    # label = image5
    return image
def convert_image(name):
    imagae = tf.io.read_file(name)
    # print(imagae)
    image = tf.image.decode_png(imagae, channels=1)  # 如果是3通道就改成3
    # print(imagae)
    image = tf.image.resize(image, [256,256])
    # image /= 255.0


    return image

#读取数据集
def load_dataset1():
    filename = os.listdir("./speckle1")
    print(filename)
    file_list = [os.path.join("./speckle1/", file) for file in filename]
    print(file_list)
    filenames = tf.constant(file_list)
    labels = tf.constant(list(range(1, 1891, 1)))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(first)
    dataset = dataset.batch(1)
    # ###序列化dataset
    # datsets = dataset.map(tf.io.serialize_tensor)
    # ###写序列化后的dataset成TF文件
    # tfrec = tf.data.experimental.TFRecordWriter('images1.tfrec')
    # tfrec.write(datsets)
    return dataset
def load_dataset2():
    filename1 = os.listdir("./speckle2")
    print(filename1)
    file_list1 = [os.path.join("./speckle2/", file) for file in filename1]
    print(file_list1)
    filenames1 = tf.constant(file_list1)
    labels1 = tf.constant(list(range(1, 211, 1)))

    dataset1 = tf.data.Dataset.from_tensor_slices((filenames1, labels1))

    dataset1 = dataset1.map(first)
    dataset1 = dataset1.batch(1)
    # print(dataset1)
    return dataset1
def load_dataset3():
    filename1 = os.listdir("./speckle3")
    print(filename1)
    # file_list1 = [os.path.join("./speckle3/", file) for file in filename1]
    file1_name = filename1 + "/1/1.png"
    file2_name = filename1 + "/1/2.png"
    file3_name = filename1 + "/1/3.png"
    file4_name = filename1 + "/1/4.png"
    # file5_name = filename + "/5.png"
    image1 = convert_image(file1_name)
    image2 = convert_image(file2_name)
    image3 = convert_image(file3_name)
    image4 = convert_image(file4_name)
    image = tf.concat([image1, image2], -1)
    image = tf.concat([image, image3], -1)
    image = tf.concat([image, image4], -1)
    # labe =  convert_image(label)
    # label = image5
    return image
def loadinput():          #这个函数是批量预测函数
    filename1 = os.listdir("./speckle7")
    file_list1 = [os.path.join("./speckle7/", file) for file in filename1]
    for x in file_list1:
        print(x)
        image1 = tf.io.read_file(x+"/1.png")
        image2 = tf.io.read_file(x+"/2.png")
        image3 = tf.io.read_file(x+"/3.png")
        image4 = tf.io.read_file(x+"/4.png")
        # print(imagae)
        image1 = tf.image.decode_png(image1, channels=1)  # 如果是3通道就改成3
        image2 = tf.image.decode_png(image2, channels=1)
        image3 = tf.image.decode_png(image3, channels=1)
        image4 = tf.image.decode_png(image4, channels=1)

        image1 = tf.image.resize(image1, [256, 256])
        image2 = tf.image.resize(image2, [256, 256])
        image3 = tf.image.resize(image3, [256, 256])
        image4 = tf.image.resize(image4, [256, 256])

        image = tf.concat([image1, image2], -1)
        image = tf.concat([image, image3], -1)
        image = tf.concat([image, image4], -1)
        image = tf.expand_dims(image, 0)
        test = model.predict(image)                 #预测预测数据集 输出为 ndarray类型

        test = test.reshape((256, 256,1))            #ndarray 来个reshape从(1，256，256，1)变为（256，256，1）

        Path = x+"/DLPU.png"

        # test1 = cv2.resize(test, (256, 256))

        cv2.imwrite(Path, test)
    return 0
def predict_input():
    image1 = tf.io.read_file("./speckle3/1/1.png")
    image2 = tf.io.read_file("./speckle3/1/2.png")
    image3 = tf.io.read_file("./speckle3/1/3.png")
    image4 = tf.io.read_file("./speckle3/1/4.png")
    # print(imagae)
    image1 = tf.image.decode_png(image1, channels=1)  # 如果是3通道就改成3
    image2 = tf.image.decode_png(image2, channels=1)
    image3 = tf.image.decode_png(image3, channels=1)
    image4 = tf.image.decode_png(image4, channels=1)

    image1 = tf.image.resize(image1, [256, 256])
    image2 = tf.image.resize(image2, [256, 256])
    image3 = tf.image.resize(image3, [256, 256])
    image4 = tf.image.resize(image4, [256, 256])

    image = tf.concat([image1, image2], -1)
    image = tf.concat([image, image3], -1)
    image = tf.concat([image, image4], -1)
    image = tf.expand_dims(image, 0)
    return image


if __name__ == '__main__':



    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # dataset = load_dataset1()        #加载训练数据集
    # dataset1 = load_dataset2()       #加载评估数据集

    image = predict_input()           #加载预测输入

    model = tf.keras.Sequential()

    model.add(layer7(0))



    model.build(input_shape=(1,256, 256,4))
    # model.summary()


    model.load_weights("./mycheckpoint06")  #加载参数
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  # 指定优化器，学习率为0.01
                  loss='mse',  # 指定均方差作为损失函数
                  metrics=['mae'])

    # model.fit(dataset,epochs=10)            #进入训练


    # loadinput()#批量预测


    # model.save_weights("./mycheckpoint05")
    # model.evaluate(dataset)                   #评论测试集

    # model.evaluate(dataset, verbose=2)
    # model.evaluate(dataset1,verbose=2)          #评论测试集




    test = model.predict(image)                 #预测预测数据集 输出为 ndarray类型


    test = test.reshape((256, 256,1))            #ndarray 来个reshape从(1，256，256，1)变为（256，256，1）

    Path = "./prediction/p.png"

    test1 = cv2.resize(test, (256, 256))


    cv2.imwrite(Path, test)

    cv2.imshow("window", test)  # 用Opencv 展示图片
    cv2.waitKey(0)

