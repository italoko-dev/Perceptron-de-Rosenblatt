import numpy as np
import copy

class Perceptron:
    def __init__(self):
        self.weights = []
        self.bias = 1
    
    #Soma Ponderada
    def net_activation(self,input):
        return  np.sum(np.multiply(input,self.weights))

    #Função Sinal
    def net_propagation(self,y):
        if y > 0 :
            return 1
        else:
            return 0

    #Função de saída da rede
    def output(self,input):
            return self.net_propagation(self.net_activation(input))
    
    #Calcula o erro da saída vs esperado
    def get_erro(self,expected,y):
        return expected - y
    
    #Ajusta os pesos
    def setup_weights(self,x,eta,erro):
        for i in range(len(self.weights)):
            self.weights[i] += eta * erro * x[i]

    #Treina a rede
    def train(self,input,eta,max_epoch):
        self.weights = np.zeros(len(input[0]),int)
        expecteds_output = copy.deepcopy(input[:, len(input[0])-1])
        input[:,len(input[0])-1] = self.bias

        for epoch in range(max_epoch):
            epoch_errors = 0
            for i in range(len(input)):
                x = input[i]
                y = self.output(x)
                error = self.get_erro(expecteds_output[i],y)
                if(error != 0):
                    self.setup_weights(x,eta,error)
                    epoch_errors += 1
            if epoch_errors == 0:
                break
        print(f'Treinado com:{epoch} epocas')
    #Predição de uma determinada entrada.
    def prediction(self,input):
        input.append(self.bias)
        print(f'Pertence à classe:{self.output(input)}')

#-----------------------------------------------

#Inputs for training 
#AND
input_train = np.array(
    [
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]          
    ])
'''
#OR
input_train = np.array(
    [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,1]          
    ])
'''
#Input for prediction 
input_prediction = [1,1]

perceptron = Perceptron()

perceptron.train(input_train,1,5)
perceptron.prediction(input_prediction)
