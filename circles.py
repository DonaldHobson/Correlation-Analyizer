import numpy as np
#colours="black,#333333,#666666,#999999,#cccccc".split(",")
colours="orange,green,blue,purple".split(",")
s=(slice(None),)
import itertools as it
def indexIterate(array,axis=None):
    if axis==None:
        axis=range(array.ndim)
    axis=set(axis)
    p=it.product(*(range(j) if i in axis else s for i,j in enumerate(array.shape)))
    for i in p:
        yield array[i]
def invert(cx,cy,cr,ix,iy,ir):
    dx=cx-ix
    dy=cy-iy
    dd=dx**2+dy**2
    #ndx=dx/dd
    #ndy=dy/dd
    ddr=dd**.5
    irs=ir**2
    #mxd=irs/(ddr-cr)
    #mnd=irs/(ddr+cr)
    inv=irs/(dd-cr**2)
    dr=cr*inv
    #dg=
    #dr=(mxd-mnd)/2
    #dg=(mxd+mnd)/2
    dx*=inv
    dy*=inv
    
    return np.concatenate(((dx+ix)[None],(dy+iy)[None],dr[None]),axis=0)#.reshape((3,-1))
transF=np.array([[1,0,-1],[0,1,-1],[1,0,1],[0,1,1]])
def inv(y,x):
    return invert(*x[:,None,:],*y[:,:,None])
def toC(x):
    return np.dot(transF,x)
import tkinter as tk
reps=2
main=tk.Tk()
c=tk.Canvas(main,background="white")
c.pack(fill="both",expand=1)

#circ=[np.array([[500,300,100],[100,253,70],[700,300,100]],float).T]#,[100,353,70],[300,353,70]],float).T]
#circ=[np.array([[ 331.,  156.,  226.],[ 195.,  214.,  371.],[ 100.,   70.,  100.]])]
circ=[np.array([[ 500.,  507.,  700.,  705.,  602.],       [ 300.,  492.,  300.,  508.,  400.],       [ 100.,   90.,  100.,  105.,   40.]])]

circleCount=circ[0].shape[1]
circSource=[np.arange(circleCount)]

for i in range(reps):
    circ.append(inv(circ[0],circ[-1])[:,circSource[-1][None,:]!=circSource[0][:,None]])
    circSource.append(np.arange(circ[-1].shape[1])//(circ[-1].shape[1]//circleCount))
    
                #[:,np.arange(circleCount**2-circleCount)+np.arange(circleCount**2-circleCount)//(circleCount)+1])#np.arange(circleCount**2)%(circleCount+1)!=0])
#for i in range(reps-1):
#    circ.append(inv(circ[0],circ[-1]))
print([i.shape for i in circ])
keys=min(circleCount,9)
circTk=[np.array([c.create_oval(*toC(i),outline=k) for i in j.T],np.object0) for j,k in zip(circ,colours)]
#circRoot=circTk[::circTk.size//(circ[0].shape[1]-1)]
#circRoot=[1,8]

circToMove=0
def redraw():
    for i in range(reps):
        circ[i+1]=inv(circ[0],circ[i])[:,circSource[i][None,:]!=circSource[0][:,None]]
    for i,j in zip(circ,circTk):
        for k,l in zip(i.T,j):
            c.coords(l,*toC(k))

def click(e=None):
    circ[0][0:2,circToMove]=e.x,e.y
    
    redraw()

c.bind('<1>',click)


def keyUp(e=None):
    circ[0][2,circToMove]+=5
    redraw()
def keyDown(e=None):
    circ[0][2,circToMove]=max(circ[0][2,circToMove]-5,15)
    redraw()

main.bind('<Up>',keyUp)
main.bind('<Down>',keyDown)
import functools as fc
def movCng(i,e=None):
        print(i,"here")
        global circToMove
        c.itemconfig(circTk[0][circToMove],outline="black")
        circToMove=i
        c.itemconfig(circTk[0][circToMove],outline="red")
movCng(0,None)
for i in range(keys):

    main.bind(str(i+1),fc.partial(movCng,i))
