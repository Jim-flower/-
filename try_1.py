import tensorflow as tf
import os
from tensorflow.keras import layers,Sequential
#残差部分
class ResiduaBlock(layers.Layer):

    def __init__(self,filter_num,strides =1):
        super(ResiduaBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=strides,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=strides,padding="same")
        self.bn2 = layers.BatchNormalization()
        if strides !=1:
            self.downsaple = Sequential()
            self.downsaple.add(layers.Conv2D(filter_num,(1,1),strides=strides))
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
#第一大层 总共分为11个大层 filter_num是通道数测试时填 8 输出为以及池化了的图片
#总共11层，左边五层用的类都一模一样 不同的是每一次的通道数不同而已即filter_num
class conv2(layers.Layer):
    def __init__(self,filter_num,strides = 1):

        super(conv2, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding="same",)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        return out
class layer1(layers.Layer):

    def __init__(self,filter_num1=8,strides =1):
        super(layer1, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第一层后\n",out)

        return out
class layer2(layers.Layer):

    def __init__(self,filter_num1=16,strides =1):
        super(layer2, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第2层后\n",out)

        return out
class layer3(layers.Layer):

    def __init__(self,filter_num1=32,strides =1):
        super(layer3, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第3层后\n",out)

        return out
class layer4(layers.Layer):

    def __init__(self,filter_num1=64,strides =1):
        super(layer4, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第4层后\n",out)

        return out
class layer5(layers.Layer):

    def __init__(self,filter_num1=128,strides =1):
        super(layer5, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第5层后\n",out)

        return out
class layer6(layers.Layer):

    def __init__(self,filter_num1=256,strides =1):
        super(layer6, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(filter_num1)
        self.l4 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                                              activation='relu')




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)#池化
        # print("经过第6层后\n",out)

        return out
class layer11(layers.Layer):

    def __init__(self,filter_num1=8,strides =1):
        super(layer11, self).__init__()
        self.l1 = conv2(filter_num1)
        self.l2 = ResiduaBlock(filter_num1)
        self.l3 = conv2(1)
        # self.l4 = layers.MaxPool2D((2,2))




    def call(self,inputs, traing =None):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        # out = self.l4(out)#池化
        # print("经过第11层后\n",out)

        return out

class layer7(layers.Layer):

    def __init__(self,filter_num1=256,strides =1):
        super(layer7, self).__init__()
        self.l1 = layer1(8)
        self.l2 = layer2()
        self.l3 = layer3()
        self.l4 = layer4()
        self.l5 = layer5()
        self.MP = layers.MaxPool2D((2,2))
        self.l6 = layer6()
        self.l7 = layer5(128)
        self.uc = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                              activation='relu')
        self.uc_1= layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                         activation='relu')

        self.l8 = layer4(64)
        self.l9 = layer3(32)

        self.uc_2 = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same',
                                           activation='relu')
        self.l10 = layer2(16)
        self.uc_3 = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',
                                           activation='relu')

        self.l11 = layer11(8)

        self.dp = layers.Dropout(rate=0.3)





    def call(self,inputs, traing =None):
        out1_1 = self.l1(inputs)
        out1 = self.MP(out1_1)
        print("out1",out1)
        # out1 = self.dp(out1)

        out2_1 = self.l2(out1)

        out2 = self.MP(out2_1)
        # out2 = self.dp(out2)

        out3_1 = self.l3(out2)
        out3 = self.MP(out3_1)

        # out3 = self.dp(out3)

        out4_1 = self.l4(out3)
        out4 = self.MP(out4_1)

        # out4 = self.dp(out4)

        out5_1 = self.l5(out4)#输出的out5没有池化
        out5 = self.MP(out5_1)

        # out5 = self.dp(out5)

        out6 = self.l6(out5)

        # out6 = self.dp(out6)

        input7 = layers.concatenate([out5_1,out6],-1)
        # input7 = layers.add([out5_1, out6])
        input7 = self.l7(input7)

        out7 = self.uc(input7)

        # out7 = self.dp(out7)

        input8 = layers.concatenate([out4_1,out7],-1)
        # input8 = layers.add([out4_1, out7])

        input8 = self.l8(input8)

        # int8 = self.uc1(input8)
        out8 = self.uc_1(input8)

        # out8 = self.dp(out8)

        input9 = layers.concatenate([out3_1,out8],-1)
        # input9 = layers.add([out3_1, out8])

        input9 = self.l9(input9)
        out9 = self.uc_2(input9)

        # out9 = self.dp(out9)

        input10 = layers.concatenate([out2_1,out9],-1)
        # input10 = layers.add([out2_1, out9])

        input10 = self.l10(input10)
        out10 = self.uc_3(input10)

        # out10 = self.dp(out10)

        input11 = layers.concatenate([out1_1, out10], -1)
        # input11 = layers.add([out1_1, out10])

        out11 = self.l11(input11)

        # out11 = self.dp(out11)



        #这里添加一个dropout
        # out11 = tf.nn.dropout(out11,rate=0.5)





        print("*******\n",out11)

        return out11


class layers_1(layers.Layer):

    def __init__(self,filter_num1,strides =1):
        super(layers_1, self).__init__()
        self.conv1 = layers.Conv2D(filter_num1, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.resblock = ResiduaBlock(filter_num1,strides =1)

        self.Maxp = layers.MaxPool2D((2,2))

    def call(self,inputs, traing =None):
        out = self.conv1(inputs)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.resblock(out)

        out = self.conv1(out)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.Maxp(out)
        return out
#这一层专门写的是第11层 输入不是池化 输出通道为1
class layers_1_1(layers.Layer):

    def __init__(self,filter_num1,strides =1):
        super(layers_1_1, self).__init__()
        self.conv1 = layers.Conv2D(filter_num1, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.resblock = ResiduaBlock(filter_num1,strides =1)

        # self.Maxp = layers.MaxPool2D
        self.conv2 = layers.Conv2D(1, (3, 3), strides=strides, padding="same")

    def call(self,inputs, traing =None):
        out = self.conv1(inputs)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.resblock(out)

        out = self.conv2(out)   #这一步里面第二次卷积通道为1

        out = self.bn1(out)

        out = self.relu(out)

        # out = self.Maxp(out)
        return out




#下面写第6层，和前五层差别很小 就在最后一步 把池化变成逆卷积 那么输出就是逆卷积的输出
class layers_6(layers.Layer):
    def __init__(self, filter_num6, strides=1):
        super(layers_6, self).__init__()
        self.conv1 = layers.Conv2D(filter_num6, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.resblock = ResiduaBlock(filter_num6, strides=1)


        # self.Maxp = layers.MaxPool2D

        self.up_conv =layers.Conv2DTranspose(filters=filter_num6, kernel_size=3,strides=2, padding='same', activation='relu')

    def call(self, inputs, traing=None):
        out = self.conv1(inputs)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.resblock(out)

        out = self.conv1(out)

        out = self.bn1(out)

        out = self.relu(out)

        # out = self.Maxp(out)

        out = self.up_conv(out)
        return out

###总路线
class layers_all(layers.Layer):
    def __init__(self,f1,f2,f3,f4,f5,f6,strides =1,input_shape=[64,64,4]):
        super(layers_all, self).__init__()
        self.lay1 = layers_1(f1)#f1是通道数
        self.lay2 = layers_1(f2)#f2是通道数
        self.lay3 = layers_1(f3)#f3是通道数
        self.lay4 = layers_1(f4)#f4是通道数
        self.lay5 = layers_1(f5)#f5是通道数
        self.lay6 = layers_1(f6)#f6是通道数

        self.lay7 = layers_6(f5)#这一层是第7层的中间部分+逆卷积 Attention!! ==输入记得要处理

        self.lay8 = layers_6(f4)#这一层是第8层的中间部分+逆卷积   Attention!! ==输入记得要处理

        self.lay9 = layers_6(f3)  # 这一层是第9层的中间部分+逆卷积    Attention!! ==输入记得要处理

        self.lay10 = layers_6(f2)  # 这一层是第10层的中间部分+逆卷积  Attention!! ==输入记得要处理

        self.lay11 = layers_1_1(f1)  # 这一层是第10层的中间部分+逆卷积  Attention!! ==输入记得要处理！！！

        # self.concatenate = layers.concatenate()

    def call(self, inputs, **kwargs):

        out1 = self.lay1(inputs)

        out2 = self.lay2(inputs)

        out3 = self.lay3(inputs)

        out4 = self.lay4(inputs)

        out5 = self.lay5(inputs)

        out6 = self.lay6(inputs)

        #url = "https://tieba.baidu.com/p/6010844179?red_tag=0090407203"
        #参照贴吧那张图对concatenation的理解
        input_7 = layers.concatenate([out5,out6],axis=-1)#

        out7 = self.lay7(input_7)

        input_8 = layers.concatenate([out7,out4],axis=-1)#

        out8 = self.lay8(input_8)

        input_9 = layers.concatenate([out8,out3],axis=-1)

        out9 = self.lay9(input_9)

        input_10 = layers.concatenate([out9,out2],axis=-1)

        out10 = self.lay10(input_10)

        input_11 = layers.concatenate([out10,out1],axis=-1)

        out = self.lay11(input_11)
        return out







