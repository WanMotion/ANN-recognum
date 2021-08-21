import numpy as np
from numpy.core.fromnumeric import reshape, shape, transpose
from numpy.core.numeric import ones
import math
class ANNModel:
    """
    三层ANN模型，输入层-隐藏层-输出层
    """
    def __init__(self,inputDimension:int,hiddenDimension:int,outputDimension:int,epoch:int,learningRate:float) -> None:
        self.inputDimension=inputDimension
        self.hiddenDimension=hiddenDimension
        self.outputDimension=outputDimension
        self.epoch=epoch
        self.learingRate=learningRate
        # xavier初始化方法
        self.W1=np.random.uniform(low=-math.sqrt(6/(self.inputDimension+self.hiddenDimension)),high=math.sqrt(6/(self.inputDimension+self.hiddenDimension)),size=(hiddenDimension,inputDimension))
        self.B1=np.random.uniform(low=-math.sqrt(6/(self.inputDimension+self.hiddenDimension)),high=math.sqrt(6/(self.inputDimension+self.hiddenDimension)),size=(hiddenDimension,1))
        self.W2=np.random.uniform(low=-math.sqrt(6/(self.outputDimension+self.hiddenDimension)),high=math.sqrt(6/(self.outputDimension+self.hiddenDimension)),size=(outputDimension,hiddenDimension))
        self.B2=np.random.uniform(low=-math.sqrt(6/(self.outputDimension+self.hiddenDimension)),high=math.sqrt(6/(self.outputDimension+self.hiddenDimension)),size=(outputDimension,1))
        # 训练集loss 和 校验集loss
        self.trainLoss=[]
        self.valLoss=[]
        # 训练集accuracy 和 校验集accuracy
        self.trainAccuracy=[]
        self.valAccuracy=[]


    def train(self,trainX,trainY,valX,valY,batch=64):
        """
        trainX为amount*inputDimension
        trainY为amount*outputDimension
        目标函数softmax交叉熵损失函数
        激活函数为sigmoid
        """
        width=trainX.shape[0]
        height=trainX.shape[1]
        if height!=self.inputDimension:
            print("输入维数错误！")
        # 按照batch进行划分
        for k in range(self.epoch):
            totalLoss=0.
            totalAccu=0.
            count=0.
            # 打乱顺序
            shuffleIndex=np.random.permutation(np.arange(width))
            trainX=trainX[shuffleIndex]
            trainY=trainY[shuffleIndex]
            for i in range(batch,width,batch):
                Xcopy=trainX[i-batch:i]
                Ycopy=trainY[i-batch:i]
                # 前向传播
                A1=np.dot(self.W1,Xcopy.T)+self.B1 # (hidden,batch)
                #print(f"A1Sum:{np.sum(A1)}")
                Z1=self.sigmoid(A1)        # (hidden,batch)
                #print(f"Z1Sum:{np.sum(Z1)}")
                A2=np.dot(self.W2,Z1)+self.B2      # (outputDimension,batch)
                Z2=self.sigmoid(A2) # (outputDimension,batch)
                # 计算目标函数，采用softmax
                Loss=self.distanceLoss(Z2.T,Ycopy)
                # Loss=self.softmaxLoss(Z2,Ycopy)
                totalLoss+=Loss
                # 后向传播 
                delta_L_Z2=self.dZ2_distance(Z2,Ycopy.T) # (outputDimension,batch)
                # delta_L_Z2=self.dZ2_softmax(Z2,Ycopy)
                delta_L_A2=delta_L_Z2*self.sigmoid(A2)*(1-self.sigmoid(A2))
                delta_L_W2=np.dot(delta_L_A2,Z1.T) # (outputDimension,hidden)
                delta_L_B2=delta_L_A2              # (outputDimension,batch)
                delta_L_A1=np.dot(self.W2.T,delta_L_A2)*self.sigmoid(A1)*(1-self.sigmoid(A1)) # (hiddenDimension,batch)
                delta_L_W1=np.dot(delta_L_A1,Xcopy) # (hiddenDimension,inputDimension)
                delta_L_B1=delta_L_A1               # (hiddenDimension,batch)
                # print(f"Variance:{np.var(A1)},{np.var(A2)}")
                # 更新参数
                self.W2=self.W2-delta_L_W2*self.learingRate
                self.B2=self.B2-np.reshape(np.average(delta_L_B2*self.learingRate,axis=1),(self.outputDimension,1))
                self.W1=self.W1-delta_L_W1*self.learingRate
                self.B1=self.B1-np.reshape(np.average(delta_L_B1*self.learingRate,axis=1),(self.hiddenDimension,1))
                # 计算准确率
                acc=self.calculateAccuracy(Z2.T,Ycopy)
                totalAccu+=(acc*batch)
                count+=batch
            self.trainLoss.append(totalLoss)
            self.trainAccuracy.append(totalAccu/count)
            # 计算此时的valLoss
            A1=np.dot(self.W1,valX.T)+self.B1
            Z1=self.sigmoid(A1)
            A2=np.dot(self.W2,Z1)+self.B2
            Z2=self.sigmoid(A2)
            # valLoss=self.distanceLoss(Z2.T,valY)
            valLoss=self.softmaxLoss(Z2,valY)
            self.valLoss.append(valLoss)
            self.valAccuracy.append(self.calculateAccuracy(Z2.T,valY))
            if k%100==0:
                print(f"epoch:{k},trainLoss:{totalLoss},valLoss:{valLoss}")

    def predict(self,testX):
        '''
        预测函数
        按照前向传播计算一次
        '''
        A1=np.dot(self.W1,np.reshape(testX,(self.inputDimension,1)))+self.B1
        Z1=self.sigmoid(A1)
        A2=np.dot(self.W2,Z1)+self.B2
        Z2=self.sigmoid(A2)
        softmaxSum=np.sum(np.exp(Z2))
        Y=np.exp(Z2)/softmaxSum
        index=np.where(Y==np.max(Y))
        return (index[0],np.max(Y))

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    # softmax 损失值计算
    def softmaxLoss(self,trainY,trueY):
        softmaxSum=np.sum(np.exp(trainY),axis=0) # (batch,)
        Y=np.exp(trainY)/softmaxSum              # (outputDimension,batch)
        loss=-np.sum(trueY[np.where(trueY>0)]*np.log(Y[np.where(trueY.T>0)]).T)
        return loss
    
    def dZ2_softmax(self,trainY,trueY):
        delta_L_Z2=trainY.copy()
        delta_L_Z2[np.where(trueY.T>0)]=delta_L_Z2[np.where(trueY.T>0)]-1 # (outputDimension,batch)
        return delta_L_Z2

    # 采用距离矢量和为损失值
    def distanceLoss(self,trainY,trueY):
        loss=0.5*np.sum((trainY-trueY)**2)
        return loss
    
    def dZ2_distance(self,trainY,trueY):
        delta_L_Z2=trainY-trueY
        return delta_L_Z2

    def calculateAccuracy(self,testY,trueY):
        '''
        计算准确率
        testY:(xx,outputDimension)
        '''
        where=np.argmax(testY,axis=1)==np.argmax(trueY,axis=1)
        return np.count_nonzero(where)/testY.shape[0]

    def validate(self,testX,testY):
        A1=np.dot(self.W1,testX.T)+self.B1
        Z1=self.sigmoid(A1)
        A2=np.dot(self.W2,Z1)+self.B2
        Z2=self.sigmoid(A2)
        return self.calculateAccuracy(Z2.T,testY)