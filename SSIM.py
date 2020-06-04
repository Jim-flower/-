from skimage.measure import compare_ssim
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
filename1 = os.listdir("./speckle7")
file_list1 = [os.path.join("./speckle7/", file) for file in filename1]
m =[]
n =[]
file = open('./data_filtered.txt', 'w')
for x in file_list1:
    s = x[11:]
    s = float(s)
    # print(s)
    m.append(s)
    img1 = imageio.imread(x+"/DLPU.png")
    y = x.replace('7','6',1)
    img2 = imageio.imread(y+"/phase.png")

    img2 = np.resize(img2,(256,256))
    img1 = np.resize(img1,(256,256))

    # print(img2.shape)
    # print(img1.shape)
    ssim = compare_ssim(img1, img2, multichannel=True)
    n.append(ssim)

    file.write(str(s)+":"+str(ssim)+"\n")

    # print(ssim)
# plt.scatter(m,n)
# plt.show()
file.close()
plt.plot(m,n)
plt.show()