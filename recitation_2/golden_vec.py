import numpy as np
from scipy.optimize import RootResults

def minimize_scalar_vec(f: callable,  bracket: tuple[float|np.ndarray], tol=1e-10):
  return root_scalar_vec(lambda x: f(x)**2)
def root_scalar_vec(f: callable, bracket: tuple[float|np.ndarray], tol=1e-10)->RootResults:
  """
  Find a root of a scalar function in a vectorized way
  
  Parameters
  ----------
  f: callable
  bracket: min and max. Iterable in the form of [min, max], where
  min and max can be either a float or an array.

  Returns
  -------
  RootResults
  """
  a = np.atleast_1d(bracket[0])
  b = np.atleast_1d(bracket[1])
  if len(a)==1:
    a = np.ones_like(f(a))*a
  if len(b)==1:
    b = np.ones_like(f(b))*b    
  alpha1=(3-np.sqrt(5))/2
  alpha2=(np.sqrt(5)-1)/2
  d=b-a
  x1=a+alpha1*d
  x2=a+alpha2*d
  f1=f(x1)**2
  f2=f(x2)**2
  xx=np.zeros_like(x1)
  d=alpha1*alpha2*d
  it=0
  while np.max(d)>tol:
    d=d*alpha2
    if2bigger=f2>f1
    i1=np.asarray(if2bigger==1).nonzero()[0]
    i2=np.asarray(if2bigger==0).nonzero()[0]
    x2[i1]=x1[i1]
    x1[i1]=x1[i1]-d[i1]
    f2[i1]=f1[i1]
    x1[i2]=x2[i2]
    x2[i2]=x2[i2]+d[i2]
    f1[i2]=f2[i2]
    xx[i1]=x1[i1]
    xx[i2]=x2[i2]
    ff=f(xx)**2
    f1[i1]=ff[i1]
    f2[i2]=ff[i2]
    it+=1
  i2less=np.asarray(f2<f1).nonzero()[0]
  x1[i2less]=x2[i2less]
  f1[i2less]=f2[i2less]
  if np.max(f(x1)**2)>1e-10:
    return RootResults(root=x1, iterations=it, function_calls=it, flag=1)
  return RootResults(root=x1, iterations=it, function_calls=it, flag=0)
