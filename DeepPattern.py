import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf

from scipy import misc
import pdb
#pdb.pm()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def show(x):
    plt.imshow(x.reshape((28,28)), cmap="Greys")
    plt.show()
#relu 0.83 acc    
#tanh 0.89

def normalize(x,axis=None):
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5
def normPlot(*x):
    for i in x:
        plt.plot(normalize(i.ravel()))
    plt.show()
gmin=np.random.normal(size=(1,100,1))*0.01
gmax=np.random.normal(size=(1,100,1))*0.01
def _deepCorrelate(data,output,dataTest,outputTest):
    data=normalize(data)[:,:,None]
    output=normalize(output)[:,None,:]
    a=Sequential()
    #a.add(Reshape((28,28,1),input_shape=(784,)))
    a.add(Conv1D(10,8,activation="relu",input_shape=data.shape[1:]))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(20,8,activation="relu"))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(output.shape[2],8))
    outputTf=tf.placeholder(tf.float32, shape=(None,)+output.shape[1:])
    #dataTf=tf.placeholder(tf.float32, shape=data.shape)
    b=tf.shape(a.output)
    #print(b)
    norm=2/tf.reduce_prod(b)
    
    error=tf.nn.l2_loss(a.output-outputTf)*tf.to_float(norm)
    
    train=tf.train.AdamOptimizer(.02).minimize(error)
    #print(a.output.shape,error)
    #print(a,a.output)
    #assert 0
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    x=np.zeros(100)


    dataTest=normalize(dataTest)[:,:,None]
    outputTest=normalize(outputTest)[:,None,:]

    
    for i in np.nditer(x, op_flags=['readwrite']):
        
        guesses=sess.run([error,train],feed_dict={a.input:data,outputTf:output})
        #if i==0:
        #    print("\t"*3,guesses[0])
        #print(guesses[0])
        guesses=sess.run([error],feed_dict={a.input:dataTest,outputTf:outputTest})

            
        i[...]=guesses[0]
    print(x[0],x[-1],x.min(),x.argmin())
    
    
    #print()
    #print(guesses[0])
    h=tf.reduce_mean(a.output)
    grad=tf.gradients(h,a.input)[0]
    global gmin,gmax
    for i in range(2):
        gmin+=sess.run([grad],feed_dict={a.input:gmin})[0]
        gmax-=sess.run([grad],feed_dict={a.input:gmax})[0]
    return x.min()
def deepCorrelate(data,output):
    x=np.zeros(10)
    for i in  np.nditer(x, op_flags=['readwrite']):
        r=np.random.random(size=data.shape[0])<0.7
        nr=np.logical_not(r)
        i[...]=_deepCorrelate(data[r],output[r],data[nr],output[nr])
        
    print(x,x.mean())
#assert 0
inpu=misc.imread("TestCorrelateData.png",flatten=True).T
outpu=np.linspace(-1,1,150)[:,None]
deepCorrelate(inpu,outpu)
assert 0
inpu=np.random.normal(size=(150,100))
outpu=np.random.normal(size=(150,3))
inpu+=outpu[:,0,None]
#_deepCorrelate(inpu[:100],outpu[:100],inpu[100:],outpu[100:])
deepCorrelate(inpu,outpu)
#print((outpu**2).sum()*6)
assert 0
