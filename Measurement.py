#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : Nathan Le Rétif

"""
Cette librairie a été crée dans le but d'automatiser la gestion et la propagation des incertitudes liées aux mesures physiques
Le coeur de la librairie utilise numpy pour la vectorisation et les performanves du C
Elle consiste à definir une serie de mesure comme un numpy.array 
"""

# Import des librairies
import matplotlib.pyplot as plt
import numpy as np


# Definition de la classe
class Measure : 

    def __init__(self, value, sigma=None, Delta=None, unit=''):

        if sigma is not None and Delta is not None : 
            raise ValueError("sigma and Delta can't be defined simulatanously")
        if sigma is None : 
            sigma = 0
        if Delta is not None : 
            sigma = Delta / np.sqrt(3)

        value = np.atleast_1d(value)
        sigma = np.atleast_1d(sigma)
        if sigma.shape != value.shape : 
            if sigma.size != 1 : 
                raise ValueError('Sigma/Delta must be scalar or the same shape as the value')
            sigma = np.ones(value.shape ,dtype=float) * sigma

        self.__value = value
        self.__sigma = sigma
        self.__unit = unit

    @property
    def value(self) : return self.__value

    @property
    def sigma(self) : return self.__sigma

    @property
    def unit(self) : return self.__unit

    
    def _round(self) : 
        zero_sigma = self.sigma==0
        safe_sigma = np.where(self.sigma==0,
                               1.,
                               self.sigma)
        odg = np.floor(np.log10(safe_sigma))
        factor = 10.0 ** odg
        rsigma = np.where(self.sigma==0,
                           0,
                           np.ceil(self.sigma/factor) * factor )
        rvalue = np.where(self.sigma==0,
                           self.value,
                           np.round(self.value/factor) * factor )
        return rvalue, rsigma, odg, zero_sigma
        

    def __repr__(self):
        return (f"Measure("
                f"value={self.value}, "
                f"sigma={self.sigma})"
                f"unit={self.unit}" )
    
    def __str__(self):
        rvalue, rsigma, odg, zero_sigma = self._round()
        rsigma = np.where(zero_sigma, 0, np.floor(rsigma*10**-odg))
        return '[' + ','.join(f'{v}({s}) '+self.unit for v,s in zip(rvalue, rsigma)) + ']'

    def __pos__(self):
        return Measure(self.value, self.sigma)

    def __neg__(self):
        return Measure(-self.value, self.sigma)

    def __add__(self, other):
        if isinstance(other, Measure) :
            value = self.value + other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value + other , self.sigma)
        return NotImplemented
    
    __radd__ = __add__ 

    def __sub__(self, other):
        if isinstance(other, Measure) :
            value = self.value - other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value - other , self.sigma)
        return NotImplemented
    
    def __rsub__(self, other):
        return Measure(other - self.value , self.sigma)
    
    def __mul__(self, other):
        if isinstance(other, Measure) :
            value = self.value * other.value
            sigma = np.sqrt( (other.value*self.sigma)**2 + (self.value*other.sigma)**2 )
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value * other, self.sigma * other)
        return NotImplemented
    
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Measure) :
            value = self.value / other.value
            sigma = np.abs(value) * np.sqrt( (self.sigma/self.value)**2 + (other.sigma/other.value)**2 )
            return Measure(value, sigma)
        if np.isscalar(other) :
            return Measure(self.value / other, self.sigma / other)
        return NotImplemented
        
    def __rtruediv__(self, other):
        if isinstance(other, Measure) :
            value = other.value / self.value
            sigma = np.sqrt( (other.sigma*self.value)**2 + (self.sigma*other.value)**2 )
            return Measure(value, sigma )
        if np.isscalar(other) :
            value = other/self.value
            sigma = np.abs(other/(self.value**2)) * self.sigma
            return Measure(value, sigma)
        return NotImplemented
    
    def __pow__(self, other):
        if np.isscalar(other) :
            value = self.value ** other
            sigma = np.abs(other*self.value**(other-1)) * self.sigma
            return Measure(value, sigma)
        return NotImplemented

    
    @staticmethod
    def propagate (func , *measures) :
        try : 
            import jax.numpy as jnp
            import jax
        except ImportError : 
            raise ImportError("To use Measure.propagate, jax library must be installed : pip install jax")
        
        values = jnp.array([m.value for m in measures])
        sigmas = jnp.array([m.sigma for m in measures])

        grad_f = jax.grad(lambda x : func(*x))
        g = grad_f(*values)

        value = func(*values)
        sigma = jnp.sqrt(jnp.sum( (g * sigmas) ** 2 ))

        return Measure(value, sigma)
    
    @staticmethod
    def errorbar (ax, x, y, **kwargs) :
        if not isinstance(x, Measure) : 
            x = Measure(x, 0)
        if not isinstance(y, Measure) : 
            y = Measure(y, 0)
        xvalue, xsigma = x._round()
        yvalue, ysigma = y._round()
        ax.errorbar(xvalue, yvalue, xerr=xsigma, yerr=ysigma, **kwargs)


    


