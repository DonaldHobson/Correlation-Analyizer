import numpy as np
import scipy as sc
import scipy.interpolate as sci
import pdb
class Dim():
    """A dimension of a data array, what a change along it represents"""
    def __init__(self,name="",size=None):
        self.name=name
        self.size=size
class SmoothDim(Dim):
    """A dimention of the array that is a discrete approximation to a smooth range of data."""
    def __init__(self,minV,maxV,name="",size=None):
        Dim.__init__(self,name,size)
        self.min=minV
        self.max=maxV
        self.range=maxV-minV
class SensorDim(Dim):
    pass
class ParameterDim(Dim):
    pass
class LocComponentsDim(Dim):
    pass
class LocLatDim(SmoothDim):
    pass
class LocLongDim(SmoothDim):
    pass
class TimeDim(SmoothDim):
    pass
#sensorSet=Dim("sensor set",20)
#parameterSet=Dim("paramator set")
#locComponents=Dim("components of location")
#locLat=smoothDim("
class Form():
    """All blocks of data have a form. A form is something that keeps track of the kind of thing
    that the data realy represents and stops you doing meaningless operations"""
    
    def __init__(self,data,dims,name=""):
        assert type(data)==np.ndarray , "Make data numpy array"
        self.data=data
        self.s=data.shape
        self.d=len(self.s)
        print(self.s)
        assert len(self.s)==len(dims),"dims must be right length"
        for i,j in zip(dims,self.s):
            i.size=j
            assert isinstance(i,Dim),"dims array must contain only Dim type objects"
        
        self.dims=dims
        
        self.name=name
    def getDimPosByClass(self,dimClass):
        d=[i for i,j in enumerate(self.dims) if isinstance(j,dimClass)]
        assert len(d)==1, "Array must have one time dimention"
        return d[0]
    def _removeDims(self,*ints):
        ints.sort(reverse=True)
        d=self.dims[:]
        for i in ints:
            d.pop(i)
        return d
    def __str__(self):
        return "%s with shape %s"%(self.name,self.data.shape)
    def averageOverTime(self,name=""):
        d=self.getDimPosByClass(TimeDim)
        return Form(np.reducemean(self.data,axis=d),self.dims[:d]+sef.dims[d+1:],name)
    
    def directCorrelate(self,form,dims,name=""):

        
        assert isinstance(form,Form)
        a=list(range(self.d))
        b=list(range(self.d,self.d+form.d))
        x=self.dims[:]
        y=form.dims[:]
        for dim in dims:
            
                
            d=self.getDimPosByClass(dim)
            e=form.getDimPosByClass(dim)
            #uncomment
            #assert x[d]==y[e]
            x[d]=None
            y[e]=None
            b[e]=a[d]

        
        return Form(np.einsum(self.data,a,form.data,b),[i for i in x if i!=None]+[i for i in y if i!=None],name)
    def getPoints(self,form):
        x=self.getDimPosByClass(LocLatDim)
        y=self.getDimPosByClass(LocLongDim)
        #assert type(sensorLocation)==SensorLocation, "invalid input type"
        z=form.getDimPosByClass(LocComponentsDim)
        
        out=np.zeros((sensorLocation.s[0],self.s[2]))
        for i in range(self.s[2]):
            f=sci.RectBivariateSpline(np.linspace(self.dims[0].min,self.dims[0].max,self.data.shape[0]),np.linspace(self.dims[1].min,self.dims[1].max,self.data.shape[1]),self.data[...,i])
            out[:,i]=f(sensorLocation.data[:,0],sensorLocation.data[:,1],grid=False)
        
        return SensorValue(out,form._removeDims(z)+self.removeDims(x,y),name="(%s) at points (%s)"%(self.name,sensorLocation.name))


class SensorValue(Form):
    """Holds a (sensorCount,parameterCount) matrix. A parameter is a single value that describes a sensor across all time,
    like an average temperature."""
    def __init__(self,data,dims,name=""):
        assert data.ndim==2, "Use 2d grid"
        Form.__init__(self,data,dims,name)
    #def directCorrelate(self,sensorValue):
    #    assert isinstance(sensorValue,SensorValue)
    #    return np.einsum("ij,ik->jk",self.data,sensorValue.data)

class SensorLocation(SensorValue):
    """Holds a (sensorCount,2) matrix. The first column is latatude, the second longdetude."""
    def __init__(self,data,dims,name=""):
        
        assert data.shape[1]==2, "each point has latitude and longditude only"
        SensorValue.__init__(self,data,dims,name)

        self.data=data
        self.name=name
    
class SensorSequence(Form):
    """Holds a (sensorCount,timesteps,parameterCount) matrix. A parameter sequence is a value that describes a sensor at a single time,
    like temperature at an instant."""
    def __init__(self,data,dims,name=""):
        Form.__init__(self,data,dims,name)

        self.data=data
        
        self.name=name
    def averageOverTime(self,name=""):
        return SensorValueSet(np.reducemean(self.data,axis=1),name)
    


class SpacialGrid(Form):
    """Holds a (xRes,yRes,parameterCount) matrix. Used to store static images or maps"""
    def __init__(self,data,dims,name=""):
        if data.ndim==2:
            data=data[...,None]
        assert data.ndim==3, "Use 3d grid"
        Form.__init__(self,data,dims,name)

        self.data=data
        self.name=name

    def getPoints(self,sensorLocation):
        #assert type(sensorLocation)==SensorLocation, "invalid input type"

        out=np.zeros((sensorLocation.s[0],self.s[2]))
        for i in range(self.s[2]):
            f=sci.RectBivariateSpline(np.linspace(self.dims[0].min,self.dims[0].max,self.data.shape[0]),np.linspace(self.dims[1].min,self.dims[1].max,self.data.shape[1]),self.data[...,i])
            out[:,i]=f(sensorLocation.data[:,0],sensorLocation.data[:,1],grid=False)
        
        return SensorValue(out,[SensorDim(),ParameterDim()],name="(%s) at points (%s)"%(self.name,sensorLocation.name))

                                


a=SpacialGrid(np.random.normal(size=(100,100,3)),[LocLatDim(-1,3),LocLongDim(-2,2),ParameterDim()])
b=Form(np.random.normal(size=(20,2)),[SensorDim(),LocComponentsDim()])
c=a.getPoints(b)
d=b.directCorrelate(c,[SensorDim])
