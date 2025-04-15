+++
title = 'L7-CNN'
date = 2024-10-29T20:20:37+08:00
draft = false
categories = ['UMich-EECS-498']
+++
# Convolutional Neural Networks

## Components of a CNN

- Convolutional layers
- Pooling layers
- Normalization layers

## Convolutional Layers

![alt text](image.png)

注意到一个通道的卷积核也是全通道数 **3** x5x5

![alt text](image-1.png)

偏置是一个向量

![alt text](image-3.png)

(b, c, h, w)表示batch size, channel, height, width!

注意四个维度的意义

卷积本质上也是一种linear layer，所以要relu等

高维全局，低维局部
![alt text](image-4.png)
### 1x1 Convolutions
![alt text](image-5.png)
一种适配器，调整通道数

### other types of convolutions
![alt text](image-6.png)
![alt text](image-7.png)
### PyTorch Implementation
![alt text](image-8.png)


## Pooling Layers
another way to downsample data, no learnable parameters

局部最大值微小移动不变性

## Normalization Layers
主要讨论的是batch normalization

层与层之间数据分布更加稳定
![alt text](image-10.png)
![alt text](image-11.png)
此时
```python
model.eval()
```
此时bn可以作为线形层被fuse进入fcnn or conv

layer norm也有，主要是rnn和transformer用到了
![alt text](image-12.png)

## Example: LeNet-5
![alt text](image-9.png)


