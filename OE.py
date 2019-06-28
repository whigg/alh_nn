#!/usr/bin/env python

'''
Optimal Estimation
==================
  
  This file implements optimal estimation in python. The method outlined in the 
  Disamar manual is followed (including pre-whitening). Disamar is of course based 
  on C. D. Rodgers' 
  U{Inverse methods for atmospheric sounding<http://www.worldscibooks.com/physics/3171.html>}).
  
  @author: Maarten Sneep
  @contact: maarten.sneep@knmi.nl
  @organization: KNMI
  @copyright: Maarten Sneep, KNMI
  @license: Creative Commons Attribution-ShareAlike 3.0 Netherlands License.
  
  The Model object
  ----------------
  
    The actual inversion is relatively simple, but data handling requires some care.
    The model, and knowlegde about its limitations are kept in a subclass of 
    the L{Model} class. The base class contains a lot of default behaviour, except 
    for the actual model itself. The model also maintains which parameters are 
    fixed, 
  
  The OE object
  -------------
    
    The L{OE} class implements optimal estimation, including pre-whitening. 
    It relies on the model for certain matrices. 
'''


import os
import sys
import time, datetime
import types
import math
import random
import warnings

try:
    from collections import OrderedDict
except ImportError:
    from OrderedDict import OrderedDict

from matplotlib import pyplot
import numpy as np

class OptimalEstimationException(Exception):
    pass
    
class DidNotConvergeWarning(Warning):
    pass
    
class Model(object):
    """The Model class for optimal estimation

The model class is a helper class for the L{optimal estimation<OE>} class. 
It maintains the state vector, prior information, physical limits, and a list of 
parameters that are fitted (or kept fixed).

This class must be subclassed and override the C{__call__} method.

Subclasses may define a "diagnostic" method which returns a dictionary with 
additional diagnostic information.

@ivar _priorCov: Internal storage of the a priori covariance matrix. Accessed as 
                 a property to check the size of the matrix, and handle a pure 
                 diagonal matrix.
@type _priorCov: A L{np.matrix} instance.
@ivar _covar:    The current covariance matrix.
@type _covar:    A L{np.matrix} instance.
@ivar _prior:    The a priori state vector.
@type _prior:    A L{np.ndarray} instance.
@ivar _statevec: Internal storage of the state vector.
@type _statevec: A L{np.ndarray} instance.
@ivar modelCalculation: The model calculation for the current state vector.
@type modelCalculation: A L{np.ndarray} instance.
@ivar Jacobian:  The Jacobian matrix, the derivative of the model calculation with 
                 respect to the state vector elements. 
@type Jacobian:  A L{np.matrix} instance.
@ivar otherModelParameters: Other parameters for the model that are not part of 
                 the state-vector, such as solar zenith angle.
@type otherModelParameters: A L{np.ndarray} instance.
@ivar parameterNames: The names of the parameters, for output purposes.
@type parameterNames: a list of strings.
@ivar stateVector: The public facing property of L{_statevec}.
@ivar priorCovariance: The public facing property of L{_covar}.
@ivar prior:     The public facing property of L{_prior}.
@ivar initialStateVector: The starting point of the fit. This may be different 
                          from the prior. the default is to use the prior.
@type initialStateVector: A L{np.ndarray} instance.
@ivar observation: The measurement. 
@type observation: A L{np.ndarray} instance with length M{n}.
@ivar observationError: The 1-sigma error on the observation.
@type observationError: A L{np.ndarray} instance with length M{n}.
@ivar independentVariable: The independent variable for the model, for our 
                           retrievals usually M{\lambda}.
@type independentVariable: A L{np.ndarray} instance with length M{n}.
@ivar verbose:   An integer indicating how chatty (and plotting) the model should be.
                 0: be silent, 1: plot to file, 2: show.
"""
    def __init__(self, prior=None, priorCov=None, 
                 otherModelParam=None, parameterNames=None, observation=None, 
                 observationError=None, independentVariable=None, 
                 stateVector=None, verbose=0, **kwargs):
        """The initializer method.

Here we check that all parameters are consistent (in length, mostly). Most 
named arguments map to instance variable in a straightforward way. 

@param prior: The value of the a priori state vector (required). This is a vector of length M{m}.
@param priorCov: The a priori covariance error matrix. The matrix can be supplied 
              as a full error covariance matrix, including relations between 
              parameters. If the parameter is supplied as a 1D-array, it is 
              considered to be the diagonal of the covariance matrix. This matrix 
              is either $M{m*m} or M{m} elements long. 
@param otherModelParam: a vector with the remaining model parameters.
@param parameterNames: The names of all parameters (statevector + parameterNames, in that order).
@param observation: The observation vector (M{B{y}}).
@param observationError: The 1-sigma error on each observation.
@param independentVariable: The wavelength scale of the observation.
@param stateVector: The initial state vector.
@param verbose: flag to indicate chatty-ness.
@param kwargs: extra keyword parameters for subclasses.
@raise ValueError: if the a priori state vecor is missing.
"""
        self._priorCov = None
        self._covar = None
        self.modelCalculation = None
        self.Jacobian = None
        self.verbose = verbose

        if prior is None:
            raise ValueError("A value for the prior statevector is required.")
            
        self._prior = prior.copy()
        self.priorCovariance = priorCov
        
        self._statevec = prior.copy()
        self.otherModelParameters = otherModelParam if otherModelParam is not None else []
        self.parameterNames = parameterNames
                
        if len(self.parameterNames) != len(self.otherModelParameters) + len(self.prior):
            m = "The number of parameters is not correct, {0} != {1} + {2}"
            raise ValueError(m.format(len(self.parameterNames), 
                                      len(self.otherModelParameters), 
                                      len(self.prior)))
        
        if stateVector is not None:
            self.stateVector = stateVector
        
        self.initialStateVector = self.stateVector.copy()
        
        self.observation = observation
        self.observationError = observationError
        self.independentVariable = independentVariable
        
        # ignore floating point errors
        np.seterr(divide='ignore', 
                  invalid='ignore', 
                  over='ignore', 
                  under='ignore')
        
    def checkObservation(self):
        """Check that the observation vectors are consistent.

The method returns a True value indicating a consistent set. The observation 
vectors are consistens if they are all available, and have the same length.

@return: C{True} if C{self.independentVariable}, C{self.observation}, and 
         C{self.observationError} are not C{None} and all have the same length.
@rtype: boolean
"""
        if (self.independentVariable is not None 
            and self.observation is not None 
            and self.observationError is not None):
            l = len(self.independentVariable)
            if (l == len(self.observation) and l == len(self.observationError)):
                return True
        return False
    
    def setObservationVector(self, independentVariable=None, 
                             observation=None, observationError=None):
        """Set the observation vector.
        
Set the complete observation vector for the Model object. That is, the 
independentVariable ('lambda'), the observation ('y') and the observation error 
('sigma_y'). These attributes can be accessed directly, but this method also
performs a consistency check, and raises a ValueError if there is a length 
mismatch.

@param independentVariable: The value for the new C{self.independentVariable} 
                            instance variable
@param observation:         The value for the new C{self.observation} instance 
                            variable
@param observationError:    The value for the new C{self.observationError} 
                            instance variable
@raise ValueError:          If L{checkObservation} fails, a ValueError is raised.
"""
        if independentVariable is not None:
            self.independentVariable = independentVariable
        if observation is not None:
            self.observation = observation
        if observationError is not None:
            self.observationError = observationError
        
        if not self.checkObservation():
            raise ValueError("The elements of the observation vector "
                "('independentVariable', 'observation' and 'observationError') "
                "are not consistent.") 
        
    def reset(self):
        self.covariance = None
        self.stateVector = self.initialStateVector
            
    def getPrior(self):
        return self._prior
    
    def setPrior(self, prior):
        if len(prior) != len(self._prior):
            m = "The number of state vector elements cannot be changed!"
            raise ValueError(m)
        self._prior = prior
    
    prior = property(getPrior, setPrior, None, 
                     "The a priori state vector for this model")
    
    def getPriorCov(self):
        return self._priorCov
    
    def setPriorCov(self, priorCov):
        if priorCov is None:
            m = "A value for the prior covariance matrix is required."
            raise ValueError(m)
        elif (len(priorCov.shape) == 1 
              and priorCov.shape[0] == len(self.prior)):
            self._priorCov = np.matrix(np.diag(priorCov**2))
        elif (len(priorCov.shape) == 2 
              and priorCov.shape[0] == priorCov.shape[1] 
              and priorCov.shape[0] == len(self.prior)):
            self._priorCov = np.matrix(priorCov.copy())
        else:
            m = """Unexpected number of dimensions of the error 
covariance matrix, or length of dimensions not 
consistent with prior."""
            raise ValueError(m)
    
    priorCovariance = property(getPriorCov, setPriorCov, None, 
                               "The a priori error covariance for this model")
    
    def getStateVector(self):
        return self._statevec
    
    def setStateVector(self, stateVector):
        if len(stateVector) != len(self._statevec):
            m = "The number of state vector elements cannot be changed!"
            raise ValueError(m)
        self._statevec = stateVector
    
    stateVector = property(getStateVector, setStateVector, None, 
                  "The state vector of the model. When setting the state vector "
                  "the values are automatically checked against the limits.")
    
    def getCovariance(self):
        return self._covar
    
    def setCovariance(self, covar):
        if covar is None:
            self._covar = None
        elif (len(covar.shape) == 1 
              and covar.shape[0] == len(self.prior)):
            self._covar = np.matrix(np.diag(covar**2))
        elif (len(covar.shape) == 2 
              and covar.shape[0] == covar.shape[1] 
              and covar.shape[0] == len(self.prior)):
            self._covar = np.matrix(covar.copy())
        else:
            raise ValueError("Unexpected number of dimensions of the error "
                             "covariance matrix, or length of dimensions not "
                             "consistent with state vector.")
    
    covariance = property(getCovariance, setCovariance, None, 
                          "The current covariance error matrix")
    
    def __str__(self):
        """informal representation"""
        m = "{name:15s}: {prior} +/- {error}; {current} +/- {curErr}; B F"
        r = [m.format(
             name="Name", prior="Prior", error="S_prior", 
             current="Current", curErr="S_curr")]
        r.append("-"*len(r[0]))
        
        m = "{name:15s}: {prior:.4g} +/- {error:.4g}; {current:.4g} +/- {curErr:.4g}"
        for i in range(len(self.stateVector)):
            r.append(m.format(
                     name=self.parameterNames[i], 
                     prior=self.prior[i], 
                     error=np.sqrt(self.priorCovariance[i,i]), 
                     current=self.stateVector[i], 
                     curErr=np.sqrt(self.covariance[i,i])))
        r.append("-"*len(r[0]))
        
        r.append("Error covariance matrix:")
        for i in range(len(self.stateVector)):
            r.append(", ".join(["{0:.3g}".format(float(v)) 
                                for v in self.covariance[:,i]]))
        r.append("-"*len(r[0]))
        
        r.append("Correlation matrix:")
        for i in range(len(self.stateVector)):
            k = []
            covii = self.covariance[i,i]
            for j in range(len(self.stateVector)):
                covjj = self.covariance[j,j]
                v = self.covariance[i,j]/(np.sqrt(covii*covjj))
                k.append("{0:.3g}".format(v))
            r.append(", ".join(k))
        r.append("-"*len(r[0]))
        
        r.append("Other parameters:")
        len_sv = len(self.stateVector)
        m = "{name:15s}: {value}"
        for i in range(len(self.stateVector), len(self.parameterNames)):
            r.append(m.format(name=self.parameterNames[i], 
                              value=self.otherModelParam[i-len_sv]))
        r.append("-"*len(r[0]))
        
        if self.verbose > 0 and self.modelCalculation is not None:
            r.append("x_indep\ty_measure\ty_error\ty_model\tdelta_y")
            s = "-"*len(r[-1])
            r.append(s)
            m = "{x:.3g}\t{y_measure:.4g}\t{y_error:.4g}\t{y_model:.4g}\t{delta:.4g}"
            for i in range(len(self.modelCalculation)):
                r.append(m.format(
                    x=self.independentVariable[i], 
                    y_measure=self.observation[i], 
                    y_error=self.observationError[i], 
                    y_model=self.modelCalculation[i], 
                    delta=((self.observation[i] - self.modelCalculation[i])/
                           self.observationError[i])))
            r.append(s)
        
        if self.verbose > 0 and self.Jacobian is not None:
            r.append("Jacobian:")
            r.append("x_indep\t" + ("\t".join(self.parameterNames[0:len_sv])))
            J = np.zeros((len(self.modelCalculation), len_sv+1))
            J[:, 0] = self.independentVariable
            J[:, 1:] = self.Jacobian
            
            for i in range(len(self.modelCalculation)):
                r.append("\t".join(["{0:.4g}".format(v) for v in J[i,:]]))
            del J
            
        return "\n".join(r)
    
    def namedModelParameter(self, s):
        """Obtain the value of a named parameter

The named parameter should be found in the L{stateVector} or in the 
L{otherModelParameters}, using the names given in the L{parameterNames}
instance variable.

@param s: name of the parameter.
@return: value of named parameter or C{None} if parameter not found.
"""
        try:
            idx = self.parameterNames.index(s)
        except ValueError:
            return None
        
        if idx >= len(self.stateVector):
            idx -= len(self.stateVector)
            val = self.otherModelParameters[idx]
        else:
            val = self.stateVector[idx]
        
        return val
    
    def indexInStateVector(self, s):
        """Obtain the index within the state vector.
        
This index can be used for building the Jacobian. Other model parameters return 
a fill value.

@param s: name of the parameter.
@return: index in state vector, or C{None} if the parameter is not in the state 
         vector.
"""
        try:
            idx = self.parameterNames.index(s)
        except ValueError:
            return None
        
        if idx < len(self.stateVector):
            return idx
        else:
            return None
        
    def __call__(self, do_jacobian=True):
        """Calculate the model with the state vector that the model-object has.

The model should be calculated at the values given by the independentVariable
parameter. This method raises a NotImplementedError, as it is expected to be 
overridden in a subclass.
"""
        raise NotImplementedError("The Model call is not implemented")
    
    def close(self):
        """Dummy method to allow subclasses to clean up after themselves."""
        pass
    
    @property
    def rmse(self):
        d = ((self.observation-self.modelCalculation)/self.observationError)**2
        return math.sqrt(np.sum(d)/len(self.observation))

    def plot(self, iteration=None, stateVectorConv=None):
        """A simple plot routine for the model."""        
        r = ["{0}".format(self.__class__.__name__)]
        if iteration is not None:
            r.append("i: {0}".format(iteration))
        fmt = lambda a : ", ".join(["{0:.4g}".format(float(v)) for v in a])
        r.append("stateVector: {0}".format(fmt(self.stateVector)))
        if stateVectorConv is not None:
            r.append("stateVectorConv: {0:.4g}".format(stateVectorConv))
            
        s = "; ".join(r)
        
        if iteration is not None and self.verbose > 0:
            print(s)
        
        if self.verbose > 4:
            nplot = 2 + len(self.stateVector)
            fig = pyplot.figure()
            fig.subplots_adjust(left=0.17, bottom=0.09, right=0.98, 
                                top=0.92, wspace=0.12, hspace=0.2)
            ax = fig.add_subplot(nplot,1,1)
            ax.set_title(s)
            ax.set_ylabel("$R [sr^{-1}]$")
            ax.plot(self.independentVariable, self.observation, 'k', 
                    label='measurement')
            ax.plot(self.independentVariable, self.modelCalculation, 'r', 
                    label='model')
            ax.legend(loc='lower right')
            
            l = fig.add_subplot(nplot,1,2)
            l.plot(self.independentVariable, 
                   (self.observation-self.modelCalculation)/self.observationError, 
                   'k', label="err")
            l.set_ylabel("$\Delta R/\sigma$")
            
            color = ['k-', 'r-', 'b-', 'g-', 'k--', 'r--', 'b--', 'g--', 'k-.', 
                     'r-.', 'b-.', 'g-.', 'k:', 'r:', 'b:', 'g:']
            for i in range(len(self.stateVector)):
                name = self.parameterNames[i]
                k = fig.add_subplot(nplot,1,3+i)
                k.plot(self.independentVariable, self.Jacobian[:, i], 'k')
                k.set_ylabel(r"$\partial R/\partial ({0})$".format(name.replace("_", " ")))
            
            k.set_xlabel("$\lambda [nm]$")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.verbose > 1:
                    fig.show()
                else:
                    fig.savefig("{0}_{1}_{2}.pdf".format(r[0], 
                                r[1].split()[1][:-1], 
                                ("{0:02d}".format(iteration) 
                                 if iteration is not None 
                                 else "final")), transparent=True)
                    
            

class OE(object):
    """The optimal estimation engine.

The optimal estimation engine uses thh model-object to obtain the measurements 
and the a priori information. 

After creating an instance, the actual fit is performed in the L{__call__} method. 

@ivar model: The C{Model} instance which performs the forward calculation and 
             holds the a priori information.
@ivar priorSinvh: One of the transformation matrices obtained with singular 
             value decomposition. This one is derived from the a priori error 
             covariance matrix. In pure latex notation: C{\mathbf{S}_a^{-1/2}}
@ivar priorSinv: One of the transformation matrices obtained with singular 
             value decomposition. This one is derived from the a priori error 
             covariance matrix. In pure latex notation: C{\mathbf{S}_a^{-1}}
@ivar priorSh: One of the transformation matrices obtained with singular value 
             decomposition. This one is derived from the a priori error 
             covariance matrix. In pure latex notation: C{\mathbf{S}_a^{1/2}}
@ivar errSinvhD: The diagonal of the inverse square root of the mearurement 
             error covariance matrix. Since the measurement error covariance 
             matrix itself is diagonal, this matrix is diagonal as well.
@ivar errSinvD: The diagonal of the inverse of the mearurement 
             error covariance matrix. Since the measurement error covariance 
             matrix itself is diagonal, this matrix is diagonal as well.
@ivar errShD: The diagonal of the square root of the mearurement 
             error covariance matrix. Since the measurement error covariance 
             matrix itself is diagonal, this matrix is diagonal as well.
@ivar U:     Result of the singular value decomposition of the Jacobian. 
             See L{w} and L{V} as well.
@ivar w:     Result of the singular value decomposition of the Jacobian. 
             See L{U} and L{V} as well.
@ivar V:     Result of the singular value decomposition of the Jacobian. 
             See L{U} and L{wV} as well.
@ivar maxiter: The maximum number of iterations.
@ivar stateVectorConvThreshold: Convergence criterium, using the change in the 
             state vector bewteen iterations as a criterium.
@ivar answer: Place to collect the diagnostic values.
"""
    def __init__(self, model=None, maxiter=12, stateVectorConvThreshold=1.0):
        """Initializer method.

The initializer method.

@param model: The model object.
@type model: an instance of L{Model}.
@param maxiter: The maximum number of iterations before giving up. Defaults to 12.
@type maxiter: integer
@param stateVectorConvThreshold: Convergence treshold, defaults to 1.0.
"""
        self.model = model
        
        self.priorSinvh = None
        self.priorSinv = None
        self.priorSh = None
        
        self.errSinvhD = None
        self.errSinvD = None
        self.errShD = None
        
        self.U = None
        self.w = None
        self.V = None
        
        self.maxiter = maxiter
        self.stateVectorConvThreshold = stateVectorConvThreshold
        
        self.answer = None
            
    def reset(self):
        self.priorSinvh = None
        self.priorSinv = None
        self.priorSh = None
        
        self.errSinvhD = None
        self.errSinvD = None
        self.errShD = None
        
        self.U = None
        self.w = None
        self.V = None
        
        self.model.reset()
        
    def start(self):
        """Set up the transformation matrices.

Use the a priori error covariance matrix and measurement errors to initialize 
the L{priorSinvh}, L{priorSinv}, L{priorSh}, L{errSinvhD}, L{errSinvD} and 
L{errShD} instance variables. These do not change between iterations.

called from the L{__call__} method.
"""
        self.model.stateVector = self.model.initialStateVector
        self.transformPriorErrorCovariance()
        self.transformMeasurementError()
    
    @property
    def Sinv(self):
        """Calculate inverse of the a posteriori error covariance matrix."""
        Wplus = np.matrix(np.diag(self.w**2 + 1.0))
        return self.priorSinvh * self.V.T * Wplus * self.V * self.priorSinvh
    
    def DecomposeJacobian(self, K):
        """Decompose the pre-whitened Jacobian using SVD.

This will fill the L{U}, L{w} and L{V} instance variables. Since these change 
with each iteration, it is important to call this method during the iteration 
sequence.

@param K: The Jacobian.
"""
        tmp = self.errSinvh * K * self.priorSh
        self.U, self.w, VT = np.linalg.linalg.svd(tmp, full_matrices=False)
        self.V = VT.T

    def calcCostFun(self):
        """ Calculate cost function and no retrievals """

        self.start()
        F, K = self.model()
        
        return self.costFunction

        
    def iterations(self):
        """Iterate to find a solution.

Run the model once for each iteration, and find a new state-vector. Continue 
until the maximum number of iterations is reached, or a solution is found 
(whichever comes first).

The method itself is described in the Disamar manual and in the support 
documentation of this code.
"""
        i = 0
        stateVectorConv = self.stateVectorConvThreshold * 1.0e6
        n = len(self.model.stateVector)
        self.answer = None
        
        while ((i < self.maxiter) 
            and (stateVectorConv > self.stateVectorConvThreshold)
            ):
            
            F, K = self.model()
            
            if np.any(np.isnan(F)) or np.any(np.isnan(K)):
                m = "Iteration {0} failure of model."
                raise OptimalEstimationException(m.format(i))
                
            if self.model.verbose > 0:
                self.model.plot(i+1, stateVectorConv)
            
            try:
                self.DecomposeJacobian(K)
            except np.linalg.LinAlgError:
                m = "Iteration {0} failure in decomposition."
                raise OptimalEstimationException(m.format(i))
            
            statevectorOffset = (self.V.T * self.priorSinvh * 
                        np.matrix(np.array(self.model.stateVector) - np.array(self.model.prior) ).T)
            measurementOffset = (self.U.T * self.errSinvh * 
                                       np.matrix(self.model.observation - F).T)
            
            newState = np.matrix((self.w * 
                              (measurementOffset.A1 + 
                              self.w * statevectorOffset.A1))/(self.w**2+1.0)).T
            newState = self.priorSh * self.V * newState
            newState = newState.A1 + self.model.prior
            
            stateVectorConv = ((np.matrix(newState - self.model.stateVector) * 
                self.Sinv * np.matrix(newState - self.model.stateVector).T)/n)[0,0]
            self.model.stateVector = newState

            if i == 0:
                
                stateVectorConv = self.stateVectorConvThreshold * 1.0e6
            
            print('cost Function for iteration {}:'.format(i), self.costFunction)

            i += 1
            
        F, K = self.model()
        if self.model.verbose > 0:
            self.model.plot(i+1, stateVectorConv)
        
        try:
            self.DecomposeJacobian(K)
        except np.linalg.LinAlgError:
            raise OptimalEstimationException("Failure in decomposition.")
        
        Wplus2 = np.matrix(np.diag(1.0/(self.w**2+1.0)))
        self.model.covariance = (self.priorSh * self.V * Wplus2 * 
                                                        self.V.T * self.priorSh)
        

        
        return i, stateVectorConv
            
    def diagnostic(self, numberOfIterations, stateVectorConv):
        """Calculate diagnostic parameters of a fit.

Calculate convergence parameters and return them in a dictionary.
The dictionary has the following keys: 'converged', 'number of iterations',
'state vector convergence criterium', 'state vector boundary violation', 
'state vector boundary violation detail', 'fixed parameter list', 
'a priori state vector', 'a priori error covariance matrix', 
'a posteriori state vector', 'a posteriori error covariance matrix',
'a posteriori noise error covariance matrix', 'cost function', 
'degrees of freedom', 'averaging kernel', 'gain matrix', and 
'weighted root mean square difference'. 
These parameters are decribed in the support documentation of this code. 

The L{answer} instance variable is set to this dictionary as well.

@param numberOfIterations: actual number of iterations.
@param stateVectorConv: state vector convergence criterium

@rtype: C{OrderedDict()}
@return: Dictionary with keys given above.
"""
        result = OrderedDict()
        result['number of iterations'] = numberOfIterations
        result['state vector element names'] = self.model.parameterNames
        result['state vector convergence criterium'] = stateVectorConv
        result['a priori state vector'] = self.model.prior
        result['a priori error covariance matrix'] = self.model.priorCovariance
        result['a posteriori state vector'] = self.model.stateVector
        result['a posteriori error covariance matrix'] = self.model.covariance
        Wplus2 = np.matrix(np.diag(self.w**2/(self.w**2+1.0)))
        AK = self.priorSh * self.V * Wplus2 * self.V.T * self.priorSinvh
        result['a posteriori noise error covariance matrix'] = AK * self.model.covariance
        result['cost function'] = self.costFunction
        result['degrees of freedom'] = AK.trace()[0,0]
        result['averaging kernel'] = AK 
        Wplus = np.matrix(np.diag(self.w/(self.w**2+1.0)))
        result['gain matrix'] = self.priorSh * self.V * Wplus * self.U.T * self.errSinvh
        result['weighted root mean square difference'] = self.model.rmse
        result['correlation matrix'] = self.correlation_matrix
        result['condition number Jacobian'] = np.linalg.cond(self.model.Jacobian)
        condition_number = np.linalg.cond(result['a posteriori error covariance matrix'])
        result['condition number a posteriori error covariance matrix'] = condition_number
        result['state vector boundary violation'] = False
        result['state vector boundary violation detail'] = None
        result['fixed parameter list'] = []

        
        result['converged'] = ((stateVectorConv < self.stateVectorConvThreshold)
                                and (numberOfIterations < self.maxiter)  )
        
        if hasattr(self.model, "diagnostic"):
            d = self.model.diagnostic()
            for k, v in list(d.items()):
                result[k] = v
        
        self.answer = result
        
        return result
    
    @property
    def correlation_matrix(self):
        """return the correlation matrix"""
        correlation_matrix = self.model.covariance.copy()
        sigmaD = np.sqrt(np.diag(correlation_matrix))
        for ii in range(correlation_matrix.shape[0]):
            for jj in range(correlation_matrix.shape[1]):
                correlation_matrix[ii, jj] /= sigmaD[ii] * sigmaD[jj]
        return correlation_matrix
        
    def __call__(self):
        """Fit the model parameters to match the data.

After creating the OE object, calling this method will perform the whole fit.

@param verbose: if C{True} create plots of the function, derivatives and 
residual for each iteration.
@rtype: C{OrderedDict()}
@return: Dictionary as returned by L{diagnostic}.
"""
        self.start()
        numberOfIterations, stateVectorConv = self.iterations()
        if numberOfIterations <= 1:
            self.answer = None
            raise DidNotConvergeWarning("Number of iterations <= 1.")
        result = self.diagnostic(numberOfIterations, stateVectorConv)
        return result
        
    def transformPriorErrorCovariance(self):
        """Set up the prior error covariance transformation matrices.

Use the a priori error covariance matrix to initialize 
the L{priorSinvh}, L{priorSinv} and L{priorSh} instance variables. 
These do not change between iterations.
"""
        U_a, w_a, V_aT = np.linalg.linalg.svd(self.model.priorCovariance, 
                                              full_matrices=False)
        V_a = V_aT.T
        self.priorSinvh = V_a * np.matrix(np.diag(np.sqrt(1.0/w_a))) * U_a.T
        self.priorSh = U_a * np.matrix(np.diag(np.sqrt(w_a))) * V_aT
        self.priorSinv = V_a * np.matrix(np.diag(1.0/w_a)) * U_a.T
    
    def transformMeasurementError(self):
        """Set up the measurement error transformation matrices.

Use the measurement errors to initialize the L{errSinvhD}, L{errSinvD} and 
L{errShD} instance variables. These do not change between iterations.
"""
        var = self.model.observationError**2
        self.errShD = self.model.observationError
        self.errSinvD = 1.0/var
        self.errSinvhD = np.sqrt(self.errSinvD)
    
    @property
    def errSinv(self):
        """Diagonal matrix, inverse measurement variance matrix"""
        return np.matrix(np.diag(self.errSinvD))
    
    @property
    def errSinvh(self):
        """Diagonal matrix, inverse measurement error matrix"""
        return np.matrix(np.diag(self.errSinvhD))
    
    @property
    def errSh(self):
        """Diagonal matrix, measurement error matrix"""
        return np.matrix(np.diag(self.errShD))
    
    @property
    def costFunction(self):
        """Calculate the cost function.

Otherwise know as the chi square value.

@return: chi square
"""
        priorDiff = np.matrix(self.model.stateVector - self.model.prior).T
        measurementDiff = np.matrix(self.model.observation
                                    - self.model.modelCalculation).T
        chisq = measurementDiff.T * self.errSinv * measurementDiff
        chisq += priorDiff.T * self.priorSinv * priorDiff
        
        return chisq[0,0]
    
    def _matrixToStr(self, name, mat):
        """Helper method to create a string representation of a matrix"""
        r = []
        r.append("\n" + name)
        for  i in range(len(self.answer['a priori state vector'])):
            r.append(", ".join(["{0:=+10.4g}".format(float(v)) 
                                for v in mat[:, i]]))
        return "\n".join(r)
        
    def __str__(self):
        if self.answer is None:
            return "None"
        r = []
        
        for k in ["converged", "number of iterations"]:
            r.append("{0}: {1}".format(k, self.answer[k]))
        
        for k in ['state vector convergence criterium', 'cost function', 
                  'degrees of freedom', 'weighted root mean square difference']:
            r.append("{0}: {1:.5g}".format(k, self.answer[k]))
        r.append("\na priori state vector:\n{0}".format(
                 ", ".join([str(float(v)) 
                            for v in self.answer['a priori state vector']])))
        
        m = self.answer['a priori error covariance matrix']
        r.append(self._matrixToStr("a priori error covariance matrix:", m))
                
        if self.answer['state vector boundary violation']:
            r.append("\nThe final state has one or more boundary violations")
            r.append("a posteriori state vector: {0}".format(
                     ", ".join([str(float(v)) 
                           for v in self.answer['a posteriori state vector']])))
            r.append("boundary violation status: {0}".format(
                     ", ".join(["{0}".format(bool(v)) 
              for v in self.answer['state vector boundary violation detail']])))
        else:
            r.append("\na posteriori state vector: {0}".format(
                     ", ".join([str(float(v)) 
                           for v in self.answer['a posteriori state vector']])))
        r.append("Fixed parameters list      : {0}".format(
                 ", ".join([str(bool(v)) 
                                for v in self.answer['fixed parameter list']])))
        
        m = self.answer['a posteriori error covariance matrix']
        r.append(self._matrixToStr("a posteriori error covariance matrix:", m))
        
        m = self.answer['a posteriori noise error covariance matrix']
        r.append(self._matrixToStr("a posteriori noise error covariance matrix:", m)) 
        
        m = self.answer['averaging kernel']
        r.append(self._matrixToStr("averaging kernel:", m))
        
        m = self.answer['gain matrix']
        r.append(self._matrixToStr("gain matrix:", m))
        
        return "\n".join(r)
        
class decay(Model):
    """Subclass of L{Model} for testing the L{OE} class

This subclass models an exponential decay with a residual offset.

y = a * exp(-x/b) + c

The derivatives of the signal with respect to the state vector parameters 
[a, b, c] are calculated analytically. 

The instance variables L{modelCalculation} and L{Jacobian} are set as well 
within the L{__call__} routine.
"""
    def __call__(self):
        """The main action takes place here. 

This part contains the model specific code. The rest of the model-code is 
used as is.
"""        
        m = np.zeros((len(self.observation),))
        k = np.zeros((len(self.observation), len(self.prior)))
        
        sv = self.stateVector
        m = sv[0] * np.exp(-(self.independentVariable/sv[1])) + sv[2]
        
        k[:, 0] = np.exp(-(self.independentVariable/sv[1]))
        k[:, 1] = (sv[0] * self.independentVariable * 
                           np.exp(-(self.independentVariable/sv[1]))/(sv[1])**2)
        k[:, 2] = np.ones((len(self.observation),))
        
        self.modelCalculation, self.Jacobian = m, k
        
        return m, k
    
    def diagnostic(self):
        d = OrderedDict()
        d['modelCalculation'] = self.modelCalculation
        d['Jacobian'] = self.Jacobian
        return d
    
def main():
    """Test the OE and Model classes"""
    prior = np.asarray([2.0, 4e-6, 1e-4])
    priorCov = np.diag(np.asarray([5.0, 1e-2, 1.0]))
    independentVariable = np.arange(0.0, 5.0e-5, 1e-7)
    
    truth = np.asarray([2.5, 1.2e-5, 4e-4])
    observation = (truth[0] * np.exp(-independentVariable/(truth[1])) + 
               truth[2] + np.random.normal(0.0, 0.1, independentVariable.shape))
    observationError = (0.1 * np.sqrt(truth[0] * 
                 np.exp(-independentVariable/(truth[1]))) + math.sqrt(truth[2]))
    
    model = decay(prior=prior, priorCov=priorCov, 
                  otherModelParam=None, 
                  parameterNames=["a", "tau", "offset"], 
                  verbose=2,
                  observation=observation, 
                  observationError=observationError,
                  independentVariable=independentVariable)
    
    oe = OE(model=model, maxiter=8)
    
    oe()
    
#    print(oe)
#    print(model)
#    
#    model.plot()
    
    return oe, model

if __name__ == "__main__":
    oe, model = main()
    
    s = input('Hit enter\n\n')
    
    model.close()
