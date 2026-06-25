#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : Nathan Le Rétif

"""
Definition of a Measure class to encapsulate the gestion and propagation of uncertainties in physics experiments, centered on a numpy architecture
Elementary operators for computation are implemented so that operations like m1 + m2 does not require heavier treatment  
More complex propagation can be made analatically through JAXpropagation method which requires the jax module or stochastically through MonteCarlo algorithm
Uncertainties are not rounded in the Measure instance, rounding is made when exporting the data (like print or plot) to not overevaluate uncertainties
Plotting through plt.errorbar is encapsulated in the Measure.errorbar method
"""

# Library import 
import matplotlib.pyplot as plt
import numpy as np

# =================== #
# Auxiliary functions #
# =================== #

def _init_error_check (value, unc, unc_type) :
    """
    Function used to handle error raising in the Measure class init method, see __init__ for arguments detail
    """
    if len(np.shape(value)) > 1 : 
        raise ValueError("value must be unidimensionnal")
    
    if unc_type in ['b','B','unchanged'] :        
        if np.any(np.asarray(unc)<0) : 
            raise ValueError("unc can't be strictly negative")
        
        elif np.size(value) != np.size(unc) and np.size(unc) > 1 : 
            raise ValueError('unc must be scalar or the same shape as value')
        
    if unc_type not in ['a','A','b','B','unchanged'] : 
        raise ValueError("uncertainty type must be 'a'/'A', 'b'/'B' or 'unchanged'")
    

def _isscalar(x) : 
    try :
        float(x)
        return True 
    except ValueError : 
        return False

def _isstring(x) :
    return isinstance(x, str) and not _isscalar(x)
    

# =============== #
# Class defintion #
# =============== #

class Measure : 

    # ======================= #
    # Instance initialisation #
    # ======================= #

    def __init__(self, value, unc=0, unc_type='B', unit = '', label=''):
        """
        Method to initialise an instance of the class
        arguments : - value : scalar or unidimensionnal array-like -> all measured values
                    - unc : scalar or array-like of same shape as value -> uncertainties of values, 
                                                                           can be scalar when unc is the same for all values,
                                                                           assumed to be 0 when undefined
                    - unc_type : 'A'/'a'/'B'/'b'/'unchanged' -> defines the uncertainty type for different treatment
                                                                'A'/'a' for type A uncertainties
                                                                'B'/'b' for type B uncertainties
                                                                'unchanged' when unc is already the correct standard deviation and does not need any treatment
        attributes: - value : float array -> all values
                    - sigma : float array of the same shape -> each corresponding standard deviation 
        """
        _init_error_check(value, unc, unc_type)

        self.label = label
        self.unit = unit

        if unc_type in ['a','A'] : 
            self.__value = np.mean(value)
            self.__sigma = np.std(value) / np.sqrt(len(value))

        elif unc_type in ['b', 'B'] : 
            self.__value = value
            self.__sigma = unc/np.sqrt(3)

        elif unc_type == 'unchanged' :
            self.__value = value
            self.__sigma = unc

        # conversion to correct size numpy array
        self.__value = np.atleast_1d(self.__value)
        self.__sigma = np.atleast_1d(self.__sigma)
        if self.__value.size > 1 and self.__sigma.size == 1 : 
            self.__sigma = np.ones(self.__value.shape) * self.__sigma


    @property
    def value(self) : return self.__value

    @property
    def sigma(self) : return self.__sigma

    @property
    def shape(self) : return self.value.shape

    @property
    def size(self) : return self.value.size

    # =================== #
    # Private methods  #
    # =================== # 
        
    def _round (self) :
        """
        private method returning a new measure instance with correct rounding
            - self.value will be rounded to the correct order of magnitude according to self.sigma
            - self.sigma will be upper rounded to the first significative digit
        Rmq : RuntimeWarning errors encountered during execution (like division by zero) are masked but do not pose any problem
        Careful for high precision measure (>=12 decimals), computation may not be exact because of floating-point precision error handling
        """
        with np.errstate(divide='ignore',invalid='ignore') : 

            # computation of correct order of magnitude
            odg = np.floor(np.log10(self.sigma)) # returns a -np.inf is self.sigma==0
            factor = 10 ** odg # returns 0. if a value is -np.inf
            mask = np.isfinite(odg)

            # rounding, np.where allows to process the above mentionned exception
            rvalue = np.where(mask, np.round(self.value/factor) * factor, self.value)
            rsigma = np.where(mask, np.ceil(self.sigma/factor), 0) * factor

            # floating-point precision error handling : 
            rvalue = np.round(rvalue,decimals=12)
            rsigma = np.round(rsigma,decimals=12)

        return Measure(rvalue, rsigma, unc_type='unchanged', unit=self.unit, label=self.label)
    
    @staticmethod
    def _string_match(line : str, sep = ',') :
        """
        match line (composed of scalars and caracter chains divided by sep) to a Measure instance according to the following rules :
        - if no scalars are found, an error will be raised
        - if exactly one scalar is found, will be assumed to be the value with no uncertainty
        - if two scalar are found, the first one is assumed to be the value and the second one to be the unc
        - if more than two scalar are found, the FIRST is assumed to be the UNIT and the rest is assumed to be all the values
        - each string will be checked to be an unc_type, order does not matter but duplicates will raise an error
        - first string not recognized as unc_type will be assumed to be the label
        - second string not recognized as unc_type will be assumed to be the unit
        - any more string in the line will raise an error 
        CAREFUL : defining a Measure with unit but no label is currently impossible 
                  parsing order for more than 2 scalars is reversed : unc,value1,value2,... instead of value,unc 
                  defining a Measure with two values or more without expliciting uncertainty is currently impossible (assumed to be value,unc or unc,value1,value2,...)
                  defining a Measure with non universal uncertainty is currently impossible
        """
        pieces = line.split(sep)
        unc_types = ['A','a','B','b','unchanged']
        kwargs = {}
        scalars = []

        for piece in pieces : 

            piece = piece.strip()
            #print(piece)

            if _isscalar(piece) : 
                scalars.append(float(piece))

            elif piece in unc_types : 
                if 'unc_type' not in kwargs : 
                    kwargs['unc_type'] = piece
                else : raise ValueError("Multiple types assigned")

            elif _isstring(piece) :
                if 'label' not in kwargs : 
                    kwargs['label'] = piece
                elif 'unit' not in kwargs : 
                    kwargs['unit'] = piece
            
            else : raise ValueError(f'Could not interpret {piece}')

        l = len(scalars)
        if l==0 : 
            raise ValueError("No numerical fields found")
        elif l==1 : 
            kwargs['value'] = scalars[0]
        elif l==2 : 
            kwargs['value'] = scalars[0]
            kwargs['unc'] = scalars[1]
        else : # when the number of numerical fields >2, assumes the first one to be the uncertainty
            kwargs['unc'] = scalars[0]
            kwargs['value'] = scalars[1:]
                
        return Measure(**kwargs)


    # =================== #
    # Elementary methods  #
    # =================== #   
    
    def __str__(self):
        """
        Method that returns the string of values under the form 'Intensity : [7.56 ± 0.04, 8.94 ± 0.02,...] mW' ; used for printing
        """
        m = self._round()
        rvalue, rsigma = m.value, m.sigma

        if np.all(rsigma==0) : # case for which there are no uncertainty (sigma=0)
            if rvalue.size == 1 : return str(rvalue[0])
            res = str(rvalue)

        elif rvalue.size == 1 : # case for which self contains only one point
            res = f'{rvalue[0]} ± {rsigma[0]}'
        
        elif np.all(rsigma==rsigma[0]) : # case for which all uncertainties are the same
            res = '[' + ','.join(str(v) for v in rvalue) + f'] ± {rsigma[0]} ' 

        else : # case for which all uncertainties are different
            res = '[' + ','.join(f'{v} ± {s}' for v,s in zip(rvalue, rsigma)) + ']'

        if self.label=='' : return res + self.unit
        else : return f"{self.label} : {res} {self.unit}"

        
    def __repr__(self):
        """
        method that returns all attributes of the class, useful for debugging
        """
        return (f"Measure("
                f"value={self.value},"
                f"sigma={self.sigma})"
                f"unit='{self.unit}')"
                f"label='{self.label}')")
    

    def __len__(self):
        return len(self.value)
    
    
    def __getitem__(self, key):
        return Measure(self.value[key], self.sigma[key], unc_type='unchanged', unit=self.unit, label=self.label)

    def __array__(self,dtype=float) :
        """
        method permetting the use of numpy functions on self directly (np.func(self)) instead of always typing np.func(self.value)
        Careful : using this will not conserve the Measure instance and all informations except self.value
        """
        return np.asarray(self.value,dtype=dtype)
    

    # ==================== #
    # Elementary operators #
    # ==================== #

    def __pos__(self):
        return Measure(self.value, self.sigma, unc_type='unchanged', unit=self.unit)

    def __neg__(self):
        return Measure(-self.value, self.sigma, unc_type='unchanged', unit=self.unit)

    def __add__(self, other):
        # defines m1 + m2 and m1 + float
        if isinstance(other, Measure) :
            value = self.value + other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma, unc_type='unchanged')
        if np.isscalar(other) :
            return Measure(self.value + other , self.sigma, unc_type='unchanged')
        return NotImplemented
    
    __radd__ = __add__ # defines float + m1

    def __sub__(self, other):
        # defines m1 - m2 and m1 - float
        if isinstance(other, Measure) :
            value = self.value - other.value
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            return Measure(value, sigma, unc_type='unchanged')
        if np.isscalar(other) :
            return Measure(self.value - other , self.sigma, unc_type='unchanged')
        return NotImplemented
    
    def __rsub__(self, other):
        # defines float - m1
        return Measure(other - self.value , self.sigma, unc_type='unchanged')
    
    def __mul__(self, other):
        # defines m1 * m2 and m1 * float
        if isinstance(other, Measure) :
            value = self.value * other.value
            sigma = np.sqrt( (other.value*self.sigma)**2 + (self.value*other.sigma)**2 )
            return Measure(value, sigma, unc_type='unchanged')
        if np.isscalar(other) :
            return Measure(self.value * other, self.sigma * np.abs(other), unc_type='unchanged')
        return NotImplemented
    
    __rmul__ = __mul__ # defines float * m1

    def __truediv__(self, other):
        # defines m1 / m2 and m1 / float
        if isinstance(other, Measure) :
            value = self.value / other.value
            sigma = np.sqrt( (self.sigma/other.value)**2 + (other.sigma*self.value/(other.value)**2)**2 )
            return Measure(value, sigma, unc_type='unchanged')
        if np.isscalar(other) :
            return Measure(self.value / other, self.sigma / np.abs(other), unc_type='unchanged')
        return NotImplemented
        
    def __rtruediv__(self, other):
        # defines float / m1
        if np.isscalar(other) :
            value = other/self.value
            sigma = np.abs(other/(self.value**2)) * self.sigma
            return Measure(value, sigma, unc_type='unchanged')
        return NotImplemented
    
    def __pow__(self, other):
        # defines m1 ** float
        if np.isscalar(other) :
            value = self.value ** other
            sigma = np.abs(value*other/self.value) * self.sigma
            return Measure(value, sigma, unc_type='unchanged')
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

    def mean (self) :
        return Measure(self.value, unc_type='A')
        
    def flip (self) : 
        return Measure(np.flip(self.value), np.flip(self.sigma), unc_type='unchanged', unit=self.unit, label=self.label)

    
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

        return Measure(value, sigma, unc_type='unchanged')
    
    # ==================================== #
    # Propagation by Monte-Carlo algorithm #
    # ==================================== #

    @staticmethod
    def MonteCarlo(estimator, *measures, N, distribution = np.random.normal) :

        if np.any(np.array([len(m) for m in measures]) != len(measures[0]) ) : 
            raise ValueError("All measures must be the same length")
        
        values = np.asarray([m.value for m in measures])
        sigmas = np.asarray([m.sigma for m in measures])

        samples = distribution(values, sigmas, size=(N, *values.shape))

        res = np.asarray([estimator(*samples[n]) for n in range(N)])

        return Measure(np.mean(res,axis=0), np.std(res,axis=0), unc_type='unchanged')
    
    # ===================================== #
    # Méthodes d'affichage avec matplotlib  #
    # ===================================== #
    
    @staticmethod
    def errorbar (ax, x, y, errors=True, set_labels=True, **kwargs) :
        """
        Static method based on plt.errorbar to plot measures and their uncertainties
        entry : - ax : instance of class Matplotlib.Axes to plot the value onto 
            - x : instance of class measure or array-like to plot on the x-axis, if not a Measure instance, assumes errorbar to be null
            - y : instance of class measure or array-like to plot on the y-axis, if not a Measure instance, assumes errorbar to be null, must be of same shape than x
            - errors : bool to indicate whether or not plotting the errorbars
            - **kwargs : special arguments to pass to the errorbar method, refer to plt.errorbar for further details
        """
        # Conversion to Measure instances
        if not isinstance(x, Measure) : 
            x = Measure(x)
        if not isinstance(y, Measure) : 
            y = Measure(y)
        x, y = x._round(), y._round()

        # plotting
        if errors : 
            ax.errorbar(x.value, y.value, xerr=x.sigma, yerr=y.sigma, **kwargs)
        else : 
            ax.plot(x.value, y.value, **kwargs)

        # Axes label setting
        if set_labels and (x.label!='' or x.unit!='') : 
            ax.set_xlabel(f"{x.label} ({x.unit})")
        if set_labels and (y.label!='' or y.unit!='') : 
            ax.set_ylabel(f"{y.label} ({y.unit})")
        

    
            

    
            


    



    


