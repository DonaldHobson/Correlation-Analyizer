import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf
import pdb


def normalize(x,axis=None):
    """returns x that has been linearly scaled so that
    x.mean(axis=axis)==zeros_like(x) and
    (x**2).mean(axis=axis)==ones_like(x)
    """
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5


def zerOne(x):
    """Returns a numpy array containing x zeros followed by x ones."""
    y=np.zeros([x*2,1],np.float32)
    y[x:]=1
    return y


def preprocess(data,cycle):
    """When called, returns cycled data followed by plain data.
    Cycling is done so that the last item is moved to the front.
    If cycle is false then neither part of the data is cycled and
    this function just returns two copies of the data.
    """
    s=data.shape[0]
    i=0
    if cycle:
        while True:
            i+=1
            if i==0: continue   #to ensure the data is permuted
            i=i%s
            p=np.arange(i,i+s)%s
            #p is a permutation of np.arange used to reorder data.
            
            yield np.concatenate([data[p],data],axis=0)
    else:
        doubled=np.concatenate([data,data],axis=0)
        while True:
            yield doubled


def makeModel(dataShape,layers):
    """ returns a keras sequential model. This must be done repeatedly
    because their is no way to reset the model to random starting values
    """
    assert len(dataShape)==2
    while True:
        model=Sequential()
        model.add(Dense(layers[0],input_shape=[1],activation="relu"))
        for i in layers[1:]:
            model.add(Dense(i,activation="relu"))
        yield model

        
class DSize():
    """Small module to hold onto any parameters that can't change between different
    pieces of dato being correlated. like size of data and how it is split.
    """
    
    def __init__(self,tr=.6):
        self.tr=tr
        self.lenTrain=None
        self.lenTest=None
        self.allOut=0
        
    def _addDataShape(self,dataShape,layers):
        self.allOut+=layers[-1]
        if self.lenTrain==self.lenTest==None:
            self.lenTrain=int(dataShape[0]*self.tr)
            self.lenTest=dataShape[0]-self.lenTrain
        else:
            assert self.lenTrain==int(dataShape[0]*self.tr)
            
    def data(self,*args,**kwargs):
        """data: a numpy ndarray of data, layers =[5,5] describes depth
        and bredth of neural net ,second:controls what correlates to
        what.
        """
        return Data(self,*args,**kwargs)

        
class Data():
    perm=None
    
    def __init__(self,dsize,data,layers=[5,5],second=False):
        self.dsize=dsize
        self.dsize._addDataShape(data.shape,layers)
        self.lenTrain=self.dsize.lenTrain
        self.lenTest=self.dsize.lenTest
        if data.ndim==1:
            data=data[:,None]
        if Data.perm is None:
            Data.perm=np.random.permutation(data.shape[0])
        data=data[Data.perm]
        data=normalize(data)
        split=self.dsize.lenTrain
        
        self.train=preprocess(data[:split],second)
        self.test=preprocess(data[split:],second)
        self.shape=data.shape
        
        
        self.model=makeModel(data.shape,layers=layers)


def correlate(*data,repeats=2,iterations=100):
    """Correlates data and output. All inputs should be 2D numpy arrays
    with the same size[0]. The output is a number between 0 and 1. 0
    represents the maximum possible correlation and 1 means no
    correlation. Repeats is how many times to run algorithm. Higher
    values will be more accurate but slower.

    The algorithm actually measures the amount of information needed to
    tell whether a pair of values are matched or not based on the
    probabilities from the neural network.
    """
    trainSeq=zerOne(data[0].lenTrain)
    testSeq=zerOne(data[0].lenTest)
    errorsNp=np.zeros([repeats,iterations])
    for thisErrors in errorsNp:#np.nditer(errorsNp, op_flags=['readwrite']):#www in range(repeats):
        print(thisErrors.shape)
        oldModels=[next(i.model) for i in data]
        oldOutputs=[i.output for i in oldModels]
        oldInputs=[i.input for i in oldModels]
        colapsed=tf.concat(oldOutputs,1)
        model=Sequential()
        model.add(Dense(20,input_shape=[int(colapsed.shape[1])],activation="relu"))
        model.add(Dense(20,activation="relu"))
        model.add(Dense(1))
        outputTf=tf.placeholder(tf.float32, shape=(None,1))
        gu=model(colapsed)
        error=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputTf,logits=gu))/np.log(2)
        train=tf.train.AdamOptimizer(.04).minimize(error)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        x=np.zeros([iterations])#iterations
        
        def trainFunc():
            for i in np.nditer(thisErrors, op_flags=['readwrite']):
                feed_dict={p:next(q.train) for p,q in zip(oldInputs,data)}
                feed_dict[outputTf]=trainSeq
                guesses=sess.run([error,train],feed_dict=feed_dict)
                print(guesses,"train")
                feed_dict={p:next(q.test) for p,q in zip(oldInputs,data)}
                feed_dict[outputTf]=testSeq
                guesses=sess.run([error],feed_dict=feed_dict)
                print(guesses)
                i[...]=guesses[0]
                yield guesses
        errorL=min(trainFunc(),key=lambda hhh:hhh[0])[0]
        #print(errorL)

    emin=errorsNp.min(1)
    
    return {"all":errorsNp, "mininum":emin, "avgmin":emin.mean(),"stdmin":emin.std()}
    
dsize=DSize(.6)
e=correlate(dsize.data(np.arange(300,dtype=np.float32)),dsize.data(np.arange(300,dtype=np.float32),second=True))

