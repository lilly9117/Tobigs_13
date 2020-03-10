import numpy as np
import math

class TwoLayerNet():
    """
    2 Layer Network를 만드려고 합니다.

    해당 네트워크는 아래의 구조를 따릅니다.

    input - Linear - ReLU - Linear - Softmax

    Softmax 결과는 입력 N개의 데이터에 대해 개별 클래스에 대한 확률입니다.
    """

    def __init__(self, X, input_size, hidden_size, output_size, std=1e-4):
         """
         네트워크에 필요한 가중치들을 initialization합니다.
         initialized by random values
         해당 가중치들은 self.params 라는 Dictionary에 담아둡니다.

         input_size: 데이터의 변수 개수 - D
         hidden_size: 히든 층의 H 개수 - H
         output_size: 클래스 개수 - C

         """

         self.params = {}
         self.params["W1"] = std * np.random.randn(input_size, hidden_size) #w1 =(d,h) 
         self.params["b1"] = np.random.randn(hidden_size) #b1 = (h,)
         self.params["W2"] = std * np.random.randn(hidden_size, output_size) #w2 = (h,c)
         self.params["b2"] = np.random.randn(output_size) #b2 = (c,)

    def forward(self, X, y=None):

        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        Linear - ReLU - Linear - Softmax - CrossEntropy Loss

        y가 주어지지 않으면 Softmax 결과 p와 Activation 결과 a를 return합니다. p와 a 모두 backward에서 미분할때 사용합니다.
        y가 주어지면 CrossEntropy Error를 return합니다.

        """

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # 여기에 p를 구하는 작업을 수행하세요.

        h = np.dot(X,W1) + b1 #linear
        a = np.maximum(h,0) #relu
        o = np.dot(a,W2) + b2 #linear
        p = np.exp(o)/np.sum(np.exp(o),axis=1).reshape(-1,1) #softmax

        if y is None: #y가 주어지지 않는 경우 - softmax 결과 p와 activation 결과 a를 return한다
            return p, a
        
        # 여기에 Loss를 구하는 작업을 수행하세요.
        
        #cross entropy loss
        #y가 주어지는 경우에는 crossentropy error을 return한다
        Loss = (-np.log(p[np.arange(N),y]).sum())/ N
#        print('loss : ',Loss)

        return Loss



    def backward(self, X, y, learning_rate=1e-5):
        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        grads에는 Loss에 대한 W1, b1, W2, b2 미분 값이 기록됩니다.

        원래 backw 미분 결과를 return 하지만
        여기서는 Gradient Descent방식으로 가중치 갱신까지 합니다.

        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N = X.shape[0] # 데이터 개수
        grads = {}

        p, a = self.forward(X)

        # 여기에 파라미터에 대한 미분을 저장하세요.

        dp = p
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if(j==y[i]):
                    dp[i][j]-=1
          # p-y #softmax 미분결과!
        da = np.heaviside(a,0) #relu 미분한 결과

        grads["W2"] = np.dot(a.T,dp) 
        grads["b2"] = np.sum(dp,axis=0)
        grads['W1'] = np.dot(X.T,np.multiply(da,np.dot(dp,W2.T)))
        grads["b1"] = np.sum(np.multiply(da,np.dot(dp,W2.T)),axis=0)
        
        #이부분 주석으로 뭐라 설명해야할지 모르겠어서 사진으로 첨부했습니당

        #learning rate곱하면서 가중치값을 backpropagation을 통해 업데이트 시킨다!
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]

    def accuracy(self, X, y):

        p, _ = self.forward(X)
        
        
        pre_p = np.argmax(p,axis=1)

        return np.sum(pre_p==y)/pre_p.shape[0]
