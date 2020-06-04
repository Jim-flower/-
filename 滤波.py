import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__author__ = "sunjingjing"


# 均值滤波
def blur(source):
    img = cv2.blur(source, (10, 10))

    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "均值滤波", (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow("blur", cv2charimg)
    return cv2charimg


# 中值滤波
def medianBlur(source):
    img = cv2.medianBlur(source, 3)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    # draw.text((0, 0), "中值滤波", (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow("medianBlur", cv2charimg)
    return cv2charimg


# 方框滤波
def BoxFilter(source):
    img = cv2.boxFilter(source, -1, (5, 5), normalize=1)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "方框滤波", (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow("boxFilter", cv2charimg)
    return cv2charimg


# 高斯滤波
def GaussianBlur(source):
    img = cv2.GaussianBlur(source, (3, 3), 0)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "高斯滤波", (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow("GaussianBlur", cv2charimg)
    return cv2charimg


# 高斯边缘检测
def Gaussian(source):
    sobelX = cv2.Sobel(source, cv2.CV_64F, 1, 0)  # x方向的梯度
    sobelY = cv2.Sobel(source, cv2.CV_64F, 0, 1)  # y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

    img = cv2.bitwise_or(sobelX, sobelY)  #
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), "高斯边缘检测", "green", font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow("GaussianBlur", cv2charimg)
    return cv2charimg


if __name__ == "__main__":
    # 加载图片
    img = cv2.imread("./speckle6/0.07//4.png")
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", img)

    img1 =blur(img)
    img2 =medianBlur(img)
    img3 =GaussianBlur(img)
    img4 = BoxFilter(img)
    # cv2.imwrite("C:\\Users\\Administrator.M8SII6QADHN3PKP\\Desktop\\p\\noise\\j\\4.png", img1)
    cv2.imwrite("C:\\Users\\Administrator.M8SII6QADHN3PKP\\Desktop\\SpeckeSuper\\speckle6\\0.4\\medianBlur4.png", img2)
    # cv2.imwrite("C:\\Users\\Administrator.M8SII6QADHN3PKP\\Desktop\\p\\noise\\g\\4.png", img3)
    # cv2.imwrite("C:\\Users\\Administrator.M8SII6QADHN3PKP\\Desktop\\p\\noise\\f\\4.png", img4)
    # Gaussian(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
