import numpy as np


leaky_relu_constant=0

def relu_scalar(n):
    if n>0:
        return n
    else:
        return n*leaky_relu_constant

def relu_scalar_prime(n):
    if n>0:
        return 1
    else:
        return leaky_relu_constant

relu=np.vectorize(relu_scalar)
reluPprime=np.vectorize(relu_scalar_prime)

sigmoid=lambda x: 1/(1+np.exp(-x))
sigmoidPrime=lambda x: np.exp(-x)/((1+np.exp(-x))**2)

def xavier_init(n_in, n_out):
    xavier_stddev = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, xavier_stddev, (n_out, n_in))

def newweights():
    w0=xavier_init(45*45, 16)
    w1=xavier_init(16, 16)
    w2=xavier_init(16, 16)
    w3=xavier_init(16, 16)
    w4=xavier_init(16, 1)
    b0=np.ones(16)*0.01
    b1=np.ones(16)*0.01
    b2=np.ones(16)*0.01
    b3=np.ones(16)*0.01
    b4=np.ones(1)*0.01
    np.save('weights-and-biases/w0.npy', w0)
    np.save('weights-and-biases/w1.npy', w1)
    np.save('weights-and-biases/w2.npy', w2)
    np.save('weights-and-biases/w3.npy', w3)
    np.save('weights-and-biases/w4.npy', w4)
    np.save('weights-and-biases/b0.npy', b0)
    np.save('weights-and-biases/b1.npy', b1)
    np.save('weights-and-biases/b2.npy', b2)
    np.save('weights-and-biases/b3.npy', b3)
    np.save('weights-and-biases/b4.npy', b4)