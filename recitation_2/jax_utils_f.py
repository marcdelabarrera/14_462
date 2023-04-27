import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional




def root_scalar_vec(f: callable, bracket: tuple, tol=1e-10)->RootResults:
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
  a = jnp.atleast_1d(bracket[0])
  b = jnp.atleast_1d(bracket[1])
  if len(a)==1 and len(b)>1:
    a = np.ones_like(len(b))*a
  elif len(b)==1 and len(a)>1:
    b = np.ones_like(len(a))*b    
  alpha1 = (3-jnp.sqrt(5))/2
  alpha2 = (jnp.sqrt(5)-1)/2
  d = b-a
  x1 = a+alpha1*d
  x2 = a+alpha2*d
  f1 = f(x1)**2
  f2 = f(x2)**2
  xx = np.zeros_like(x1)
  d = alpha1*alpha2*d
  it = 0
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






import numpy as np
def goldsvec(fun,bracket,tol=1e-10):
    '''
    minimize
    f is a vectorized scalar function
    bracket: lower and upper bounds of the minimization
    '''

    a,b=bracket
    alpha1=(3-np.sqrt(5))/2
    alpha2=(np.sqrt(5)-1)/2
    d=b-a
    x1=a+alpha1*d
    x2=a+alpha2*d
    f1=fun(x1)
    f2=fun(x2)
    xx=0*x1
    d=alpha1*alpha2*d
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
        ff=fun(xx)
        f1[i1]=ff[i1]
        f2[i2]=ff[i2]
        i2less=np.asarray(f2<f1).nonzero()[0]
        x1[i2less]=x2[i2less]
        f1[i2less]=f2[i2less]
    return x1,f1




@jax.jit
def interp2d(
    x:Array,
    y:Array,
    xp:Array,
    yp:Array,
    zp:Array,
    fill_value:Optional[Array] = None,
    search_method:Optional[str] = 'scan',
    ) -> Array:
    """
    Performs two dimensional lienar interpolation given two grids xp and yp and 
    a matrix of values zp at a new points x and y returning result z
    
    Source: https://github.com/adam-coogan/jaxinterp2d/blob/98810756e69ffe0162defa8815383f858c57233a/src/jaxinterp2d/__init__.py
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
        fill_value: value to fill in `z` for cases where x or y out of bounds
            set for xp and yp
        search_method: One of ‘scan’ (default) or ‘sort’. Controls the method used by the 
            jnp.searchsorted. ‘scan’ tends to be more performant on CPU, while ‘sort’ is 
            often more performant on accelerator backends like GPU and TPU.
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    # Grid search
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right", method=search_method), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right", method=search_method), 1, len(yp) - 1)

    # Select grid points
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    # Interpolation formula: https://en.wikipedia.org/wiki/Bilinear_interpolation
    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    # Fill out-of-bounds points
    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(
                x > xp[-1], jnp.logical_or(
                    y < yp[0], y > yp[-1]
                    )
                )
            )
        z = jnp.where(oob, fill_value, z)

    return z