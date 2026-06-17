#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : Nathan Le Rétif


# Library import 
import matplotlib.pyplot as plt
import numpy as np

# =============== #
# Class defintion #
# =============== #

class Measure : 

    # ======================= #
    # Instance initialisation #
    # ======================= #

    def __init__(self, value, Delta=None ,sigma=None, type='B'):

        if type=='a' or type=='A' : 

            # Private attribute assignement
            self.__value = np.mean(value)
            self.__sigma = np.std(value)/np.sqrt(len(value))

        elif type=='b' or type=='B' :

            # Parametrisation of sigma
            if sigma is not None and Delta is not None : raise ValueError("sigma and Delta can't be defined simulatanously")
            elif sigma is None and Delta is None : sigma = 0
            elif Delta is not None : sigma = Delta / np.sqrt(3)

            # Conversion to numpy arrays
            value = np.atleast_1d(value)
            sigma = np.atleast_1d(sigma)
            if sigma.shape != value.shape : 
                if sigma.size != 1 : 
                    raise ValueError('Sigma/Delta must be scalar or the same shape as the value')
                sigma = np.ones(value.shape) * sigma

            # Exception when a value of sigma is negative
            if np.any(sigma<0) : 
                raise ValueError("sigma can't be strictly negative")

            # Private attribute assignement
            self.__value = value
            self.__sigma = sigma
        
        else : raise ValueError('The uncertainty type must be defined (either A or B)')
        

    # =================== #
    # Elementary methods  #
    # =================== #

    @property
    def value(self) : return self.__value

    @property
    def sigma(self) : return self.__sigma

    @property
    def shape(self) : return self.value.shape

    @property
    def size(self) : return self.value.size

    
    def _format_data(self) : 
        """
        Private method rounding the uncertainties to the superior and rounding the values to the correct order of magnitude 
        returns : - array rvalue corresponding to the rounded values
            - array rsigma_int corresponding to the integer of the rounded sigmas, used for representation in __repr__
            - array rsigma_dec corresponding to the rounded sigmas, used for plotting in errorbar
        Rmq : RuntimeWarning errors encountered during execution (like division by zero) are masked but do not pose any problem
        """
        with np.errstate(divide='ignore',invalid='ignore') : 

            # computation of correct order of magnitude
            odg = np.floor(np.log10(self.sigma)) # returns a -np.inf is self.sigma==0
            factor = 10 ** odg # returns 0. if a value is -np.inf
            mask = np.isfinite(odg)

            # rounding, np.where allows to process the above mentionned exception
            rvalue = np.where(
                mask,
                np.round(self.value/factor) * factor,
                self.value   )
            rsigma_int = np.where(
                mask,
                np.ceil(self.sigma/factor),
                0                ).astype(int)

        rsigma_dec = rsigma_int * factor

        return rvalue, rsigma_int, rsigma_dec
        

    def __repr__(self):
        # returns all attribute of the class, useful for debugging
        return (f"Measure("
                f"value={self.value}, "
                f"sigma={self.sigma})")
    

    def __str__(self):
        # Method that returns the string of values under the form [7.56(4), 8.94(2),...] ; used for printing
        rvalue, rsigma_int, _ = self._format_data()
        if np.all(rsigma_int==0) :
            if rvalue.size == 1 : return str(rvalue[0])
            else : return str(rvalue)

        if rvalue.size == 1 : 
            return  f'{rvalue[0]}({rsigma_int[0]})'
        return '[' + ','.join(f'{v}({s})' for v,s in zip(rvalue, rsigma_int)) + ']'
    
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, key):
        return Measure(self.value[key], self.sigma[key])
    

    # ==================== #
    # Elementary operators #
    # ==================== #

    def __pos__(self):
        return Measure(self.value, self.sigma)

    def __neg__(self):
        return Measure(-self.value, self.sigma)

    def __add__(self, other):
        # defines m1 + m2 and m1 + float
        if isinstance(other, Measure) :
            value = self.value + other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value + other , self.sigma)
        return NotImplemented
    
    __radd__ = __add__ # defines float + m1

    def __sub__(self, other):
        # defines m1 - m2 and m1 - float
        if isinstance(other, Measure) :
            value = self.value - other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value - other , self.sigma)
        return NotImplemented
    
    def __rsub__(self, other):
        # defines float - m1
        return Measure(other - self.value , self.sigma)
    
    def __mul__(self, other):
        # defines m1 * m2 and m1 * float
        if isinstance(other, Measure) :
            value = self.value * other.value
            sigma = np.sqrt( (other.value*self.sigma)**2 + (self.value*other.sigma)**2 )
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value * other, self.sigma * np.abs(other))
        return NotImplemented
    
    __rmul__ = __mul__ # defines float * m1

    def __truediv__(self, other):
        # defines m1 / m2 and m1 / float
        if isinstance(other, Measure) :
            value = self.value / other.value
            sigma = np.sqrt( (self.sigma/other.value)**2 + (other.sigma*self.value/(other.value)**2)**2 )
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value / other, self.sigma / np.abs(other))
        return NotImplemented
        
    def __rtruediv__(self, other):
        # defines float / m1
        if np.isscalar(other) :
            value = other/self.value
            sigma = np.abs(other/(self.value**2)) * self.sigma
            return Measure(value, sigma)
        return NotImplemented
    
    def __pow__(self, other):
        # defines m1 ** float
        if np.isscalar(other) :
            value = self.value ** other
            sigma = np.abs(other*self.value**(other-1)) * self.sigma
            return Measure(value, sigma)
        return NotImplemented
    
    # ===================== #
    # Non trivial operators #
    # ===================== #

    def argmax(self) :
        return np.argmax(self.value)

    def max(self) : 
        return self[self.argmax()]
    
    def argmin(self) :
        return np.argmin(self.value)
    
    def min(self) : 
        return self[self.argmin()]

    def mean (self, type='std') :
        avg = np.mean(self.value)
        if type=='std' :
            return Measure(avg, np.std(self.value,mean=avg)/np.sqrt(len(self)))
        elif type=='sigma' : 
            return Measure(avg, np.sum(self.sigma**2)/len(self))
        else : 
            raise ValueError('Uncertainty computation type must be defined (either std or sigma)')

    
    # ================================ #
    # Propagation in complex functions #
    # ================================ #

    @staticmethod
    def JAXpropagate (func, *measures, **kwargs) :
        """
        Static method allowing the propagation of uncertainties through complex fonction
        Use JAX library to compute gradients : jax must be installed (pip install jax)
        entry : - func : function to propagate, must initially be a scalar function, vectorized with jax.numpy base functions (for example f = lambda x,y : jnp.sqrt(x*y))
            - *measures : array-like of instances of measure [m1,...,mN] such that y = func(m1,...,mN)
            - **kwargs : special arguments to pass to jax.grad
        returns : - an instance of Measure (y,u_y) such that y = func(m1,...,mN) and associated propagated uncertainties
        """
        # Check if jax library is installed
        try : 
            import jax.numpy as jnp
            from jax import grad, vmap
        except ImportError : 
            raise ImportError("To use Measure.propagate, jax library must be installed")
        
        # Check if Measure instances have the same shape
        try : 
            np.shape(measures) # raise an error if parts are inhomogeneous
        except ValueError: 
            raise ValueError('All meausure instances must be the same shape')

        values = [ jnp.asarray(m.value,dtype=float) for m in measures ]
        sigmas = [ jnp.asarray(m.sigma,dtype=float) for m in measures ]

        grad_f = grad(func, argnums= tuple(range(len(values))), **kwargs)
        map_grad_f = vmap(grad_f)
        g = jnp.asarray(map_grad_f(*values),dtype=float)

        value = func(*values)
        sigma = jnp.sqrt(
            jnp.sum((g*sigmas) ** 2,
                    axis=0) )

        return Measure(value, sigma)
    
    # ===================================== #
    # Méthodes d'affichage avec matplotlib  #
    # ===================================== #
    
    @staticmethod
    def errorbar (ax, x, y, errors=True, **kwargs) :
        """
        Static method based on plt.errorbar to plot measures and their uncertainties
        entry : - ax : instance of class Matplotlib.Axes to plot the value onto 
            - x : instance of class measure or array-like to plot on the x-axis, if not a Measure instance, assumes errorbar to be null
            - y : instance of class measure or array-like to plot on the y-axis, if not a Measure instance, assumes errorbar to be null, must be of same shape than x
            - errors : bool to indicate whether or not plotting the errorbars
            - **kwargs : special arguments to pass to the errorbar method, refer to plt.errorbar for further details
        """
        if not isinstance(x, Measure) : 
            x = Measure(x)
        if not isinstance(y, Measure) : 
            y = Measure(y)
        xvalue, _, xsigma = x._format_data()
        yvalue, _, ysigma = y._format_data()
        if errors : 
            ax.errorbar(xvalue, yvalue, xerr=xsigma, yerr=ysigma, **kwargs)
        else : 
            ax.plot(xvalue, yvalue, **kwargs)


            


    



    


