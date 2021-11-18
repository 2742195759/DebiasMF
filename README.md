# DebiasMF
This is the implement of MF, MF-IPS and MF-GAN

# 中文描述

这个仓库有4个文件，分别是MF，MF-IPS，MF-GAN和MF-Weight
其中MF是原始的MF实现，MF-IPS是Inverse-Propensity的实现。MF-GAN是使用GAN的方式对weights进行学习，
训练完毕之后会有一个MF-GAN/cache/weights.ascii 的文件作为权重文件，然后运行MF-Weight就会读取权重进行训练。

# 依赖库

本仓库依赖 修改之后的 cvpods 库：https://github.com/2742195759/cvpods. 可以使用docker进行安装：`docker push 2742195759/xkpods:v1.0` 安装完毕之后记得在/root/cvpods下进行pull并
重新 python setup.py install 一下。

# 安装数据集

使用docker之后，可以在/home/data/dataset/目录下解压缩下述文件：

# 

