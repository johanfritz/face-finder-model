from simple_functions import *

class Network:
    def __init__(self):
        self.w0=np.load('weights-and-biases/w0.npy')
        self.w1=np.load('weights-and-biases/w1.npy')
        self.w2=np.load('weights-and-biases/w2.npy')
        self.w3=np.load('weights-and-biases/w3.npy')
        self.w4=np.load('weights-and-biases/w4.npy')
        self.b0=np.load('weights-and-biases/b0.npy')
        self.b1=np.load('weights-and-biases/b1.npy')
        self.b2=np.load('weights-and-biases/b2.npy')
        self.b3=np.load('weights-and-biases/b3.npy')
        self.b4=np.load('weights-and-biases/b4.npy')
    def save(self):
        np.save('weights-and-biases/w0.npy', self.w0)
        np.save('weights-and-biases/w1.npy', self.w1)
        np.save('weights-and-biases/w2.npy', self.w2)
        np.save('weights-and-biases/w3.npy', self.w3)
        np.save('weights-and-biases/w4.npy', self.w4)
        np.save('weights-and-biases/b0.npy', self.b0)
        np.save('weights-and-biases/b1.npy', self.b1)
        np.save('weights-and-biases/b2.npy', self.b2)
        np.save('weights-and-biases/b3.npy', self.b3)
        np.save('weights-and-biases/b4.npy', self.b4)
    