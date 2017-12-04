import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf
import pdb

def normalize(x,axis=None):
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5
def zerOne(x):
    y=np.zeros([x*2,1],np.float32)
    y[x:]=1
    return y
def preprocess(data,second):
    data=normalize(data)
    s=data.shape[0]
    i=0
    #print(data.ndim)
    while True:
        #print(second)
        #input(i)
        i+=second
        if not i and second: continue
        
        p=np.arange(i,i+s)%s
        
        yield np.concatenate([data[p],data],axis=0)

    
def makeModel(dataShape,layers):
    assert len(dataShape)==2
    print("Done")
    while True:
        model=Sequential()
        model.add(Dense(layers[0],input_shape=[1],activation="relu"))#[dataShape[1]*2]))
        for i in layers[1:]:
            model.add(Dense(i,activation="relu"))
        yield model
                        
class Data():
    perm=None
    def __init__(self,data,layers=[5,5],datatype="none",second=False,tr=.6):
        if data.ndim==1:
            data=data[:,None]
        #np.random.seed(123098)
        if Data.perm==None:
            Data.perm=np.random.permutation(data.shape[0])
        #print(data.shape)
        data=data[Data.perm]
        
        #np.random.shuffle(data)
        split=int(data.shape[0]*tr)
        
        self.train=preprocess(data[:split],second)
        self.test=preprocess(data[split:],second)
        self.shape=data.shape
        self.lenTrain=split
        self.lenTest=data.shape[0]-split
        self.preprocess=preprocess(data,second)
        self.model=makeModel(data.shape,layers=layers)
        
            
def correlate(*data):
    trainSeq=zerOne(data[0].lenTrain)
    testSeq=zerOne(data[0].lenTest)
    for www in range(1):
        model=Sequential()
        #t=[next(i.model).output for i in data]
        #o=[next(i.model).input for i in data]
        v=[next(i.model) for i in data]
        t=[i.output for i in v]
        o=[i.input for i in v]
        colapsed=tf.concat(t,1)
        #model.add(Concatenate(axis=1,input_shape=[sum(i.shape[1] for i in data)])(t))
        model.add(Dense(20,input_shape=[int(colapsed.shape[1])],activation="relu"))
        model.add(Dense(20,activation="relu"))
        model.add(Dense(1))
        outputTf=tf.placeholder(tf.float32, shape=(None,1))
        gu=model(colapsed)
        #error=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputTf,logits=gu))
        error=tf.reduce_mean((outputTf-gu)**2)
        train=tf.train.AdamOptimizer(.04).minimize(error)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        x=np.zeros([93])#iterations
        
        #isReal=np.zeros([data.lenTrain*2,1],np.float32)
        #isReal[data[0].lenTrain:]=1
            
        def trainFunc():
            for i in np.nditer(x, op_flags=['readwrite']):
                #shift=np.random.randint(1,data.shape[0])
                feed_dict={p:next(q.train) for p,q in zip(o,data)}
                
                feed_dict[outputTf]=trainSeq
##                for n in feed_dict:
##                    print(n,feed_dict[n].T,feed_dict[n].shape,"\n")
##                assert 0
                #print(feed_dict[o[0]]==feed_dict[o[1]])
                #assert 0
                guesses=sess.run([error,train],feed_dict=feed_dict)
                print(guesses,"train")

                
                feed_dict={p:next(q.test) for p,q in zip(o,data)}
                feed_dict[outputTf]=testSeq
  
                guesses=sess.run([error],feed_dict=feed_dict)
                print(guesses)
                #guesses=sess.run([error,NNlogErr],feed_dict={a.input:dataTest,outputTf:outputTest})
                i[...]=guesses[0]
                
                yield guesses
            
        errorL=min(trainFunc(),key=lambda hhh:hhh[0])[0]
        print(errorL)
        assert 0
    print(model,model.output)
    
correlate(Data(np.arange(300,dtype=np.float32)),Data(np.arange(300,dtype=np.float32),second=True))
        
