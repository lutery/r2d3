from keras.layers import *

class ImageModel():
    """ Abstract base class for all implemented ImageModel. """
    # 图像特征提取模型的基类
    def create_image_model(self, c, enable_lstm):
        raise NotImplementedError()
    

class DQNImageModel(ImageModel):
    """ native dqn image model
    https://arxiv.org/abs/1312.5602
    根据该基类
    """

    def create_image_model(self, c, enable_lstm):
        '''
        Docstring for create_image_model
        
        :param self: Description
        :param c: 输入的数据，可以理解为一个张量
        :param enable_lstm: 是否启用了LSTM
        '''
        
        if enable_lstm:
            # 原来这里的LSTM是使用了TimeDistributed来处理的
            # TimeDistributed 会对每个时间步独立应用相同的层，并且共享权重
            # 相当于在每个时间步上应用相同的卷积层
            c = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same"), name="conv_1")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same"), name="conv_2")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same"), name="conv_3")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Flatten())(c)

        else:
            # 否则就是卷棘直接提取特征
            c = Conv2D(32, (8, 8), strides=(4, 4), padding="same", name="conv_1")(c)
            c = Activation("relu")(c)

            c = Conv2D(64, (4, 4), strides=(2, 2), padding="same", name="conv_2")(c)
            c = Activation("relu")(c)

            c = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv_3")(c)
            c = Activation("relu")(c)

            c = Flatten()(c)

        return c


