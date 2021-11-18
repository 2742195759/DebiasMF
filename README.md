# DebiasMF
This is the implement of MF, MF-IPS and MF-GAN

# 中文描述

这个仓库有4个文件，分别是MF，MF-IPS，MF-GAN和MF-Weight
其中MF是原始的MF实现，MF-IPS是Inverse-Propensity的实现。MF-GAN是使用GAN的方式对weights进行学习，
训练完毕之后会有一个MF-GAN/cache/weights.ascii 的文件作为权重文件，然后运行MF-Weight就会读取权重进行训练。

# 依赖库

本仓库依赖 修改之后的 cvpods 库：https://github.com/2742195759/cvpods. 可以使用docker进行安装：`docker push 2742195759/xkpods:v1.0` 安装完毕之后记得在/root/cvpods下进行pull并
重新 python setup.py install 一下。

# docker安装更新依赖库

使用docker安装的话，需要更新cvpods库，进入~/cvpods，git pull 之后运行 python setup.py install。即可

# 安装数据集

使用docker之后，可以在/home/data/dataset/目录下解压缩下述文件：https://drive.google.com/file/d/1puKeDekgSxJ1Z7puW3dG9_rI2un2DJ9h/view?usp=sharing 
最后会在 dataset 下生成一个 rec_debias 文件，如此，coat 和 yahoo 数据集将可以使用。

# 子文件的启动

除了MF-GAN，每个子文件都是一个cvpods项目，启动方式很简单：安装完毕cvpods之后，执行`pods_train --num-gpus 1` 即可。

对于MF-GAN，启动方式 `python gan_train.py` 每个迭代之后都会更新 MF-GAN/cache/weights.ascii 文件。运行完毕之后进入MF-Weights运行 `pods_train --num-gpus 1` 即可。

# 注意事项

对于一些问题，比如目录找不到可以先看看是不是绝对路径（需要按照自己放置位置进行修正），或者是需要创建目录（自行创建即可，比如Output之类的输出，还有cache之类的输出）
