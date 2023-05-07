from simple_functions import *

class Face_classification_model:
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
        self.iterations=0
        self.gradientw0=np.zeros(np.shape(self.w0))
        self.gradientw1=np.zeros(np.shape(self.w1))
        self.gradientw2=np.zeros(np.shape(self.w2))
        self.gradientw3=np.zeros(np.shape(self.w3))
        self.gradientw4=np.zeros(np.shape(self.w4))
        self.gradientb0=np.zeros(np.shape(self.b0))
        self.gradientb1=np.zeros(np.shape(self.b1))
        self.gradientb2=np.zeros(np.shape(self.b2))
        self.gradientb3=np.zeros(np.shape(self.b3))
        self.gradientb4=np.zeros(np.shape(self.b4))
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
    def update(self):
        if self.iterations!=0:
            self.w0-=self.gradientw0/self.iterations
            self.w1-=self.gradientw1/self.iterations
            self.w2-=self.gradientw2/self.iterations
            self.w3-=self.gradientw3/self.iterations
            self.w4-=self.gradientw4/self.iterations
            self.b0-=self.gradientb0/self.iterations
            self.b1-=self.gradientb1/self.iterations
            self.b2-=self.gradientb2/self.iterations
            self.b3-=self.gradientb3/self.iterations
            self.b4-=self.gradientb4/self.iterations
            self.iterations=0
    def descent(self, image, face, learningrate):
        cv=np.zeros(2)
        image, a1, a2, a3, a4, a5, z1, z2, z3, z4, z5=self.activations(image)
        if face:
            cv[0]=1
        z1prime=reluPrime(z1)
        z2prime=reluPrime(z2)
        z3prime=reluPrime(z3)
        z4prime=reluPrime(z4)
        z5prime=sigmoidPrime(z5)
        delta1=a5-cv
        delta1*=2*z5prime #delta1: gradb4
        delta2=delta1.reshape(np.shape(self.b4))*a4 #delta2: gradw4
        delta3=np.dot(self.w4.T, delta1)*z4prime #delta3: gradb3
        delta4=delta3.reshape(np.shape(self.b3))*a3 #delta4: gradw3
        delta5=np.dot(self.w3.T, delta3)*z3prime #delta5: gradb2
        delta6=delta5.reshape(np.shape(self.b2))*a2 #delta6: gradw2
        delta7=np.dot(self.w2.T, delta5)*z2prime #delta7: gradb1
        delta8=delta7.reshape(np.shape(self.b1))*a1 #delta8: gradw1
        delta9=np.dot(self.w1.T, delta7)*z1prime
        delta10=delta9.reshape(np.shape(self.b0))*image
        self.gw0+=delta10*learningrate
        self.gw1+=delta8*learningrate
        self.gw2+=delta6*learningrate
        self.gw3+=delta4*learningrate
        self.gw4+=delta2*learningrate
        self.gb0+=delta9*learningrate
        self.gb1+=delta7*learningrate
        self.gb2+=delta5*learningrate
        self.gb3+=delta3*learningrate
        self.gb4+=delta1*learningrate
        self.iterations+=1
        return None
    def activations(self, image):
        #image=cv2.Canny(image, 350, 250, 80)
        if np.ndim(image)!=1:
            image=image.reshape((175*150))
        if np.amax(image)>2:
            image=normalize(image)
        z1=np.matmul(self.w0, image) + self.b0
        a1=relu(z1)
        z2=np.matmul(self.w1, a1)+self.b1
        a2=relu(z2)
        z3=np.matmul(self.w2, a2)+self.b2
        a3=relu(z3)
        z4=np.matmul(self.w3, a3)+self.b3
        a4=relu(z4)
        z5=np.matmul(self.w4,a4)+self.b4
        a5=sigmoid(z5)
        return image, a1, a2, a3, a4, a5, z1, z2, z3, z4, z5