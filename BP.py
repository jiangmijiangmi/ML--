import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0-tanh(x)**2

class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
        if activation=='logistic':
            self.activation=logistic
            self.activation_deriv=logistic_derivative

        elif activation=='tanh':
            self.activation=tanh
            self.activation_deriv=tanh_deriv

        self.weights=[]

        for i in range (1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)

            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)


    def fit(self,X,y,learning_rate=0.1,epochs=10000):
        X=np.atleast_2d(X)
        temp=np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1]=X
        X=temp
        y=np.array(y)

        #意思是随机梯度下降，所以在每个epoch就随机选择一组数据作为输入

        for k in range(epochs):
            i=np.random.randint(X.shape[0])
            a=[X[i]]


        #这里的意思是把每一层的输出都存下来，然后最后一层就是最终输出
            for l in range (len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))

            error=y[i]-a[-1]
            deltas=[error*self.activation_deriv(a[-1])]
            if k % 1000 == 0:
                print(k, '...', error * error * 100)

            #残差等于=激励函数的导数*上层传来的误差

            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            #上层传来的误差=上层残差的加权和

            deltas.reverse()

            for i in range(len(self.weights)):
                layer=np.atleast_2d(a[i])
                delta=np.atleast_2d(deltas[i])
                self.weights[i]+=learning_rate*layer.T.dot(delta)

    def predict(self,x):
        x=np.array(x)
        temp=np.ones(x.shape[0]+1)
        temp[0:-1]=x
        a=temp
        for i in range(0,len(self.weights)):
            a=self.activation(np.dot(a,self.weights[i]))

        return a


nn=NeuralNetwork([2,2,1],'tanh')

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.predict(i))


