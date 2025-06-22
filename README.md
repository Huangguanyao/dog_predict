一个基于CNN模型的宠物狗识别

数据集采用的是十种类别的宠物狗，分别为beagle, dachshund, dalmatian, jindo, maltese,pomeranian, retriever, ShihTzu,   toypoodle,  Yorkshirerrier。
每种类别的宠物狗的图片大概在300张，将这300张中，90%分为训练集，10%为测试集。 

两种数据读取方式，一种简单的data_load,第二种data_load2:对数据集进行了增强，进行了随机裁剪+缩放，50%概率水平翻转，颜色扰动。

模型的搭建 

第一种模型，普通的CNN模型较为简单，采用第一种数据读取，测试集的最高准确率在40%上下

第二种模型，在普通的CNN模型上，每个卷积层后添加批归一化，加入Dropout 正则化，卷积核尺寸从5x5改为了3x3，特征通道增长。
           模型二采用第一种数据读取，测试集的准确率可达52%
           模型二采用第二种数据读取，测试集的准确率可达62%

第三种模型，在CNN的基础上加上了注意力机制，采用第二种数据读取方式，准确率可以高达79%


数据集放在archive中，三种模型在model中，训练函数为main.py，predict.py为预测函数，预测单张图片。


![img](https://github.com/user-attachments/assets/5b9531cc-3abc-4da1-8a84-5c0784d660cc)
![img_1](https://github.com/user-attachments/assets/a1178690-cad3-4a60-ac2e-158f273d3e52)
![img_2](https://github.com/user-attachments/assets/7ac770d0-ce75-4c8a-a708-0b0fd5195fe6)
