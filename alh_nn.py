#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:25:45 2018

@author: nanda
"""

""" IMPORTS """

import sys, os, tables, glob, collections, time, itertools, copy
#sys.path.insert(0, '/nobackup/users/nanda/Projects/ALH_NN/s5p___alh_nn/py3ALH')
#sys.path.insert(0, '/nobackup/users/nanda/Projects/ALH_NN/s5p___alh_nn')
import numpy as np
import netCDF4 as nc
#from time import gmtime, strftime
import convolution__module
import convol_resample, extractsurfacealbedo, obtainTempPresProfile_fromMeteoData, readTROPOMIradIrr
from OE import OE, Model
from scipy import interpolate
import tensorflow as tf

def flatten(x):
    """ flattens a list of lists to a single list """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

#%%

"""  

The following classes import and run isolated tensorflow models (or 'graphs')
The objective is to initialise the trained tensorflow model (from directory)
into a class ONCE, and run the model based on model input parameters. This way
the overhead of model initialisation is reduced to once per model call, instead
of once per model run.

"""

class import_model_ap():
    
    """  
    
    The following imports the tensorflow model for a priori (hence, 'ap') alh
    and aot. This is separate from the forward model ('fr') as the tensorflow
    model input parameters are different for either model types.
    
    PRIMARY issues one may face while porting to C++:
        1. many implementations may be python programming environment specific.
        2. https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305
    
    """
    def __init__(self, loc, y_len):
        
        """
        Initialises tensorflow model from:
            'loc' : location of tensorflow model's path.
            'y_len' : length of the prediction parameter: in this case, it's 1.
                
        """
        # Create local graph and use it in the session
        
        self.graph = tf.Graph()                     # initialises a tensorflow 'graph'
        self.sess = tf.Session(graph=self.graph)    # initialises a tensorflow session
        self.inp = None                             # input parameters to predict output 'y'
        self.output = None                          # output parameter 'y'
        self.y_len = y_len                          # length of 'y'

        # initialisation for feature scaling parameters
        # a neural network is trained on a data set, which is an mxn matrix, m 
        # being the number of data and n being the number of parameters in the model.
        # To ensure that the neural network optimises faster and better, the
        # input data has to be scaled. Due to this, all inputs must be scaled
        # exactly according to the scaling values for each parameters
        
        self.deScaled = None                        # de-scaled output parameter 'y'
        self.scale = None                           # scaling parameter 1
        self.offset = None                          # scaling parameter 2
        self.feature_scaled_inp = None              # scaled input parameters to predict 'y'
        
        self.modelDir = loc                         
        
        # the following loads the names of the model's feature vector (x parameters 
        # that are used to predict y). This is necessary, as the 'graph' stores 
        # input parameters names and, once called, requires the user to input 
        # the parameters in the exact same sequence as it was trained in.
        
        # test_pred*.h5 is a file that contains the column names as well as scaling parameters etc.
        
        with tables.open_file(glob.glob('{}/test_pred*.h5'.format(self.modelDir))[0], 'r') as f:
            
            self.cols = f.root.column_names[0:-self.y_len]
            
        # Feature vector cols are converted into a format that tensorflow accepts
        # following the code here: 
        # https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column
        # I decode self.cols because, for some reason, pytables stores strings as
        # binary strings in python3.
        
        self.feature_cols = [tf.feature_column.numeric_column(k.decode()) for k in self.cols ]
                
        # most of what is below is a property of the tensorflow model itself.
        # the estimator chosen is DNNRegressor, which is the deep learning
        # suite of tensorflow. The neural network architecture chosen is
        # 2 hidden layers between the input and the output layer, each containing
        # 100 and 50 units. The optimiser is non-relevant, because it is a feature
        # of the training code. But I am unable to import the model without mentioning
        # the optimiser that I chose. the parameter:
        #           'label_dimension' 
        # is basically the length of the output layer. In this case, it is either
        # alh or aot. Also, 'with self.graph.as_default()' is a way of opening
        # the graph, saving the model into the memory and closing the graph.
        
        """ import model from path into self.trained_model """
        
        with self.graph.as_default():
            
            # Import saved model from location 'loc' into local graph
            
            self.trained_model = tf.estimator.DNNRegressor(feature_columns=self.feature_cols,
                                        hidden_units=[100, 50], 
                                        optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.01, beta1=0.9,
                                                      beta2=0.999, epsilon=1e-08,),
                                        label_dimension=self.y_len, model_dir=self.modelDir)
                                        
    def run(self, inp):

        """
        Runs the model for the input parameters specified in:
           'inp'
        The input parameters are not feature scaled. Without feature scaling,
        the model output will be nonsensical. To avoid this, the input is feature
        scaled automatically using the feature scaling constants stored in the
        model directory. The module that does this is self.feature_scaling().
        
        """
        self.inp = inp          # store input into self
        self.feature_scaling()  # scale input 
        
        # predict the model output. the input_fn is basically an input function
        # that provides the tensorflow model with a tensorflow-supported format
        # for predictions.
        
        predictions = self.trained_model.predict( input_fn=get_input_fn_predict(self.feature_scaled_inp,
                                                [k.decode() for k in self.cols] ) )

        len_inp = 1             # length of input parameters (n input parameters, m x n inputs, m is len_inp)
        
        # itertools.islice:
        # Make an iterator that returns selected elements from the iterable. 
        # If start is non-zero, then elements from the iterable are skipped until 
        # start is reached. Afterward, elements are returned consecutively unless 
        # step is set higher than one which results in items being skipped.
        
        self.output = list(p["predictions"] for p in itertools.islice(predictions, len_inp))[0]
            
        
        # once output is found, de-scale it back to human-level perception using
        # the same constants from the model directory.
        
        self.feature_deScaling()
        
        return self.deScaled

    def feature_scaling(self):
        
        """
        Scales input parameters according to model's scaling and offset constants
        The constants are in the file test_predictions' in the model directory.
        
        These constants are derived from the training set. Possible that application
        to alien input parameters (outside of training set) could be a liability.
        Unsure.
        
        """
        with tables.open_file(glob.glob('{}/test_pred*.h5'.format(self.modelDir))[0], 'r') as f:
            
            self.scale = f.root.scale[:]
            self.offset = f.root.offset[:]

        if type(self.scale) is list:
            self.scale = np.array(self.scale)
            self.offset = np.array(self.offset)

        # scaling the input parameters:
        
        self.feature_scaled_inp = (self.inp - self.offset[:-self.y_len])/self.scale[:-self.y_len]
        
    def feature_deScaling(self):
        """
        De-scales the output from the model to real values that humans can understand.
        
        """
        
        self.deScaled = (self.output * self.scale[-self.y_len:]) + self.offset[-self.y_len:]


class import_model_fr():
    """  
    
    The following imports the tensorflow model for forward model (hence, 'fr'). 
    This is separate from the a priori model ('ap') as the tensorflow model input
    parameters are different for either model types.
    
    PRIMARY issues one may face while porting to C++:
        1. many implementations may be python programming environment specific.
    
    label_dimension = 3980 since thats the number of output wavelengths at which
    forward model calculations are done.
    
    """
    
    def __init__(self, loc, config, act_fn):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.modelDir = loc
        self.inp = None
        self.output = None
        self.deScaled = None
        self.scale = None
        self.offset = None
        self.feature_scaled_inp = None
        
        files = glob.glob('{}/test_pred*.h5'.format(self.modelDir))
        files = files[0]
        with tables.open_file(files, 'r') as f:
            
            x_len=0
            y_len=0
            for i in f.root.column_names:
                if not i.decode().startswith('y'):
                    x_len+=1
                else:
                    y_len+=1
                    
            self.x_len = x_len
            self.cols = f.root.column_names[0:self.x_len]
            
        # Feature cols
        feature_cols = [tf.feature_column.numeric_column(k.decode()) for k in self.cols]
        with self.graph.as_default():
            
            # Import saved model from location 'loc' into local graph
            self.trained_model = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=config, activation_fn = act_fn,
                                        optimizer=tf.train.AdamOptimizer(
                                                      learning_rate=0.0001, beta1=0.9,
                                                      beta2=0.999, epsilon=1e-08,),
                                        label_dimension=y_len, model_dir=self.modelDir)

    def run(self, inp):

        self.inp = inp
        self.feature_scaling()
        
        predictions = self.trained_model.predict( input_fn=get_input_fn_predict(self.feature_scaled_inp,
                                                [k.decode() for k in self.cols] ) )
        
        self.output = list(p["predictions"] for p in itertools.islice(predictions, 1))[0]
        
        self.feature_deScaling()
        
        return self.deScaled


    def feature_scaling(self):
        
        with tables.open_file(glob.glob('{}/test_pred*.h5'.format(self.modelDir))[0], 'r') as f:
            
            self.scale = f.root.scale[:]
            self.offset = f.root.offset[:]
        
        if type(self.scale) is list:
            self.scale = np.array(self.scale)
            self.offset = np.array(self.offset)
        
        self.feature_scaled_inp = (self.inp - self.offset[0:self.x_len])/self.scale[0:self.x_len]
        

    def feature_deScaling(self):
        
        self.deScaled = (np.array(self.output) * self.scale[-1]) + self.offset[-1]

def get_input_fn_predict(data_set, features):
    
    """
    sets input parameters in tensorflow-supported format
    
    """
    
    x = { features[i] : np.array([data_set[i]]) for i in range(len(data_set)) }
    
    return tf.estimator.inputs.numpy_input_fn(x = x,
                                              y = None, 
                                              num_epochs = None, 
                                              shuffle = False)

#%%
    
class aerosolLayerHeight(Model):
    
    def __init__(self, prior=None, priorCov=None, limits=None, otherModelParam=None, 
                       parameterNames=None, observation=None, observationError=None,
                       independentVariable=None, stateVector=None, initialStateVector=None, keepFixed=None, 
                       verbose=False, **kwargs):
        
        super(self.__class__, self).__init__(prior=prior, priorCov=priorCov, limits=limits, 
              otherModelParam=otherModelParam, parameterNames=parameterNames, observation=observation, 
              observationError=observationError, independentVariable=independentVariable, stateVector=stateVector, 
              keepFixed=keepFixed, verbose=verbose)
        
        
        self.minOptThicknessAer = 1E-2      # minimum aerosol optical thickness is limited to 0.01
        self.maxOptThicknessAer = 15.0      # maximum aerosol optical thickness is limited to 15.0
        
        # this keeps track of whether the state vector is adjusted or not:
        self.adjustedStateVector = False    # if the state vector is modified 2 times in a row, OE is cancelled
        # the OE module has been modified in accordance to the above
        
        self.guessApriori = kwargs['guessApriori']      # choose whether to guess apriori or not
        self.models = kwargs['models']
        self.modelsAP = kwargs['modelsAP']
        self.geo = kwargs['geo']
        self.modelPath = kwargs['modelPath']
        self.modelPathAP = kwargs['modelPathAP']
        self.fitparams = kwargs['fitparams']       
        self.pxl = kwargs['pxl']
        self.dynamicScaling = kwargs['dynamicScaling']

        """ 
        NN model input parameters may include the following (depending on ap 
        or fr model): 
            
            'saa'  : solar azimuth angle in radians
            'vaa'  : viewing azimuth angle in radians
            'mu'   : cosine of viewing zenith angle
            'mu0'  : cosine of solat zenith angle
            'psfc' : surface pressure
            'temp' : interpolant of the temperature-pressure profile
            'z_aer': input alh
            'aot'  : input aot
            'asfc' : surface albedo at 758 nm
            
        in array 'inp'
        
        """

        self.saa        = np.deg2rad(self.geo[-2])
        self.vaa        = np.deg2rad(self.geo[-1])
        self.mu         = kwargs['mu']
        self.mu0        = kwargs['mu0'] 
        self.psfc       = kwargs['psfc']
        self.asfc       = kwargs['asfc']
        self.temp       = kwargs['temp']        # INTERPOLANT. NOT FLOAT
        self.HRwavel    = kwargs['HRwav']
        self.Gweights   = kwargs['gaussian_weights']
        
        # if the prior value is guessed from NN
        if self.guessApriori:
            self.initialStateVector = self.guess_apriori()
                
        """ import NN forward models """
        
        self.NN_I_lambda = import_model_fr(os.path.join(self.modelPath, self.models[0]), [100, 100], tf.nn.sigmoid)    # for sun-normalised radiance
        self.NN_K_zaer   = import_model_fr(os.path.join(self.modelPath, self.models[1]), [100, 100], tf.nn.sigmoid)    # for derivative w.r.t alh
        self.NN_K_tau    = import_model_fr(os.path.join(self.modelPath, self.models[2]), [100, 100], tf.nn.sigmoid)    # for derivative w.r.t aot

        if self.dynamicScaling:
            
            # derivatives w.r.t surface albedo
            self.NN_K_as_758 = import_model_fr(os.path.join(self.modelPath, self.models[3]), [100, 200], tf.nn.relu)
            self.NN_K_as_770 = import_model_fr(os.path.join(self.modelPath, self.models[4]), [100, 200], tf.nn.relu)
        
    def guess_apriori(self):
        
        """ This module guesses a-priori values for the aot and alh for the optimal estimation """
        
        # first convolve the input observations for NN apriori model's accepted
        # spectral resolution. By doing this, we are maintaining a constant wavelength
        # grid that does not fluctuate from pixel to pixel.
        
        WAVEL_START = 759.0
        WAVEL_END = 769.0
        DWAVEL = 0.20
        FWHM = 0.20
        
        wavel = np.arange(WAVEL_START, WAVEL_END + DWAVEL, DWAVEL)
        conversion_factor = self.mu0 / np.pi                        # converts reflectance to sun-normalised radiance
        # the above is important since the NN model is trained with sun-normalised radiances. Although this is a minor
        # issue, since we can also train the NN apriori model to reflectances instead.

        # convol_resample code is provided by Pepijn Veefkind. self.independentVariable
        # is the input wavelength grid from instrument observation. This code basically 
        # convolves the observation to a different spectral resolution:
        
        res   =  convol_resample.convol_resample(self.independentVariable, self.observation,
                                                                 wavel, fwhm=FWHM, k=4.)
        
        # convert reflectance to sun normalised radiance :
        
        self.ap_inpspectra = (np.asarray(res[0]) * np.asscalar(conversion_factor) ).tolist()
        
        """ import a-priori NN models """
        
        self.NN_ap__zaer = import_model_ap(os.path.join(self.modelPathAP, self.modelsAP[0]),1)
        self.NN_ap__aot  = import_model_ap(os.path.join(self.modelPathAP, self.modelsAP[1]),1)

        """ guess apriori values """

        # compile input parameters into a single flattened list. the input parameters
        # are: solar azimuth angle in radians, viewing azimuth angle in radians, 
        # cosine of viewing zenith angle, cosine of solar zenith angle, surface pressure,
        # temperature at surface, temperature at 300 hPa, and temperature at 500 hPa.
        
        # Because so many different temperatures are required, it is a good practice
        # to have the temperature profile interpolant as an input, instead of different
        # numbers. Whether this is a preferred choice for the level-2 processor developer
        # is something that they can consider.

        inp_ap = flatten([self.saa, self.vaa, self.mu,self.mu0,self.psfc, self.asfc,
                               self.ap_inpspectra, 
                               [self.temp(self.psfc).tolist(), 
                                self.temp(300).tolist(), self.temp(500).tolist()
                                ] ] )
                
        prior = [list(self.NN_ap__zaer.run(inp_ap))[0], 
                      list(self.NN_ap__aot.run(inp_ap))[0]]

        print('NN derived apriori for alh: ', prior[0], 'hPa or ' )
        print('NN derived apriori for aot: ', prior[1], '[-]')
        
        # check if input alh is too close to the surface. this is a check built 
        # into and borrowed from disamar:
        
        if np.abs(prior[0] + 25.0 / self.psfc) < 0.99900:
            prior[0] -= 50.0
            print('first guess of aerosol bottom layer is too close to the surface.')
            print('adjusting bottom layer above surface pressure by 25 hPa')
        
        return prior

    def scale_height(self):
        
        """ 
        scales derivatives w.r.t. alh (in meters) to alh (in hPa), since the NN
        model predicts derivatives of reflectance w.r.t alh in m, not in hPa.
        
        """
        R = 8.31451 # [J K-1 mole-1] universal gas constant
        T = self.temp(self.z_aer) # temperature at ALH
        M = 28.964 # [kg mole-1] mean molecular mass of dry air
        g = 9.81   # [m s-2] gravitational acceleration
        
        H = R*T/(g*M)
        
        return H

    def descale_hPa(self, val):
        
        """
        converts retrieved alh in hPa to km
        unsure if correct ... 
        
        """

        R = 8.31451 # [J K-1 mole-1] universal gas constant
        T = self.temp(val) # temperature at ALH
        M = 28.964 # [kg mole-1] mean molecular mass of dry air
        g = 9.81   # [m s-2] gravitational acceleration
        
        H = R*T/(g*M)
        
        return -H*np.log(val/self.psfc)

#%% main CALL of the function

    def __call__(self, do_Jacobian=True):
        
        """ initialise forward model with named model parameters"""
        
        iter__z_aer   = self.namedModelParameter('z_aer')
        bottom_layer_hPa    = iter__z_aer + 25.0 # hPa
        top_layer_hPa       = iter__z_aer - 25.0 # hPa
        
        self.z_aer = iter__z_aer
        
        if (bottom_layer_hPa / self.psfc > 0.99900):
            print('aerosol bottom layer is too close to the surface.')
            print('adjusting bottom layer above surface pressure by 25 hPa')
            self.z_aer = self.psfc - 50.0
        
        """ 
        
        adjust alh in case of crossing boundary conditions within OE
        if alh bottom is below psfc (surface pressure), the bring alh mid pressure
        to (psfc - 200) hPa
        
        if alh top is above TOA, do the exact same as above (psfc - 200)
        
        """
        
        if bottom_layer_hPa > self.psfc:
            
            self.z_aer = self.psfc - 200
            
            print('alh is below surface; adjusting alh mid to within boundary conditions to {}'.format(self.z_aer))
            self.adjustedStateVector = True

        if top_layer_hPa < 0.3 :

            self.z_aer = self.psfc - 200
            
            print('alh is above top of atmosphere; adjusting alh mid to within boundary conditions to {}'.format(self.z_aer))
            self.adjustedStateVector = True
        

        """ adjust aot in case of crossing boundary conditions within OE"""

        val = self.namedModelParameter('aot')
        
        if ( val > self.minOptThicknessAer ) and ( val < self.maxOptThicknessAer ):
            
            val = val  
            
        else:
            
            if val < self.minOptThicknessAer:
    
                val = 0.5
                print('aot less than min aot value of {0}; adjusting aot within boundary conditions to {1}'
                      .format(self.minOptThicknessAer, val))
                self.adjustedStateVector = True
    
            if val > self.maxOptThicknessAer:
                
                val = 0.5
                print('aot greater than min aot value of {0}; adjusting aot within boundary conditions to {1}'
                      .format(self.maxOptThicknessAer, val))
                self.adjustedStateVector = True
                
        self.aot = val
                
        """ input for forward model """
        
        # for the convolution:
        wav_start = 758.0
        wav_end = 770.0
        
        # input parameters for the forward model include:
        # solar azimuth angle in radians, viewing azimuth angle in radians, 
        # cosine of viewing zenith angle, cosine of solar zenith angle, surface pressure,
        # temperature at aerosol layer height (in hPa), aerosol layer height (in hPa),
        # aerosol optical thickness, and surface albedo.
        
        inp_fr = [self.saa,self.vaa,self.mu,self.mu0,self.psfc,self.temp(self.z_aer),self.z_aer,
                                              self.aot,self.asfc]
        
        """ run forward model and convolve """
        
        # predicts hi-resolution sun-normalised radiance for the given input parameters
        
        HR_sun_normalised_radiance = self.NN_I_lambda.run(inp_fr)   
        HR_I = HR_sun_normalised_radiance * np.pi/self.mu0      # convert sun-normalised radiance to reflectance
        
        # convolution of the hi resolution sun-normalised radiance to the same at instrument grid:
        # in python, this step currently takes more time that to compute forward model outputs.
        
        wavl_final, self.modelCalculation = convolution__module.convolve_isrf(wav_start, wav_end, self.pxl, 
                                         HR_I, self.HRwavel,
                                         self.independentVariable, self.Gweights)
        
        # predicts and convolves the derivative of reflectance w.r.t alh. These derivatives
        # are converted to hPa space from meters space, using self.scale_height() and the formula:
        # K_zaer(hPa) = -K_zaer(km)*self.scale_height() / z_aer(hPa)
        
        wavl_final, k_zaer = convolution__module.convolve_isrf(wav_start, wav_end, self.pxl, 
                        -(self.scale_height()*self.NN_K_zaer.run(inp_fr)/self.z_aer), self.HRwavel,
                                             self.independentVariable, self.Gweights)


        # predicts and convolves derivative of reflectance w.r.t aot
        wavl_final, k_aot = convolution__module.convolve_isrf(wav_start, wav_end, self.pxl, 
                                             self.NN_K_tau.run(inp_fr), self.HRwavel,
                                             self.independentVariable, self.Gweights)
        
        self.Jacobian = np.zeros( ( len(k_aot), len(self.fitparams) ) )
        self.Jacobian[:,0] = k_zaer
        self.Jacobian[:,1] = k_aot
        

        if self.dynamicScaling:
            
            wavl_final, k_as758  = convolution__module.convolve_isrf(wav_start, wav_end, self.pxl, 
                                                 self.NN_K_as_758.run(inp_fr), self.HRwavel,
                                                 self.independentVariable, self.Gweights)
            
            wavl_final, k_as770  = convolution__module.convolve_isrf(wav_start, wav_end, self.pxl, 
                                                 self.NN_K_as_770.run(inp_fr), self.HRwavel,
                                                 self.independentVariable, self.Gweights)
            
            k_as = 0.5*(k_as758 + k_as770)
            M = np.abs(k_as/self.Jacobian[:,0])
            f = np.abs(k_as/self.Jacobian[:,1])
            threshold = np.percentile(M, 15)
            logical_ds = M > threshold
            self.observationError[logical_ds] *= f[logical_ds]
            self.dynamicScaling = False

        return self.modelCalculation, self.Jacobian      

#%%
        
class NNoutput(tables.IsDescription):
    
    """
    
    py tables initialisation. unimportant for C++
    
    """
    
    crlat = tables.Float32Col(shape=(4,))
    crlon = tables.Float32Col(shape=(4,))
    clat  = tables.Float32Col()
    clon  = tables.Float32Col()
    scl= tables.Int32Col()
    pxl = tables.Int32Col()
    asfc = tables.Float32Col()
    
    ret_alh = tables.Float32Col()
    ret_alh_km = tables.Float32Col()
    ret_aot = tables.Float32Col()
    apriori_alh = tables.Float32Col()
    apriori_alh_km = tables.Float32Col()
    apriori_aot = tables.Float32Col()
    
    wrmse = tables.Float32Col()
    chi2 = tables.Float32Col()
    dof = tables.Float32Col()
    aPosErrCovMat = tables.Float32Col(shape = (2,2))
    numiter = tables.Float32Col()
    svConvCrit = tables.Float32Col()
    converged = tables.StringCol(16)
    oeTime = tables.Float32Col()
    
def main():

    orbit = 4361
    orbit_path = './orbits/{}'.format(orbit)

    with tables.open_file('./s5_0107594.h5', 
                              'r') as f:
            
        weights = f.root.additional_output_sim.HR_weights_band_1[:]
        t_wavel = f.root.additional_output_sim.HR_wavelength_band_1[:]
        
    filenameRad = glob.glob(os.path.join(orbit_path,'S5P_OFFL_L1B_RA_BD6*.nc'))[0]
    try:
        filenameIrr = glob.glob(os.path.join(orbit_path,'S5P_TEST_L1B_IR*.nc'))[0]
    except Exception as e:
        filenameIrr = glob.glob(os.path.join(orbit_path,'S5P_OFFL_L1B_IR*.nc'))[0]
    meteofile       = glob.glob(os.path.join(orbit_path,'S5P_OFFL_AUX_MET_TP*.nc'))[0]
    meteofile_sp    = glob.glob(os.path.join(orbit_path,'S5P_OFFL_AUX_MET_2D*.nc'))[0]
    LER = os.path.join('./', 'GOME-2_MetOp-A_MSC_025x025_surface_LER_v2.1.hdf5')
    month = int(filenameRad.split('_')[-1][4:6])
    tropl2out = glob.glob(os.path.join(orbit_path,'S5P_TEST_L2__AER_LH*.nc'))[0]
    
    with tables.open_file( LER, 'r' ) as f:
        
        dataLat = f.root.Latitude[:]
        dataLon = f.root.Longitude[:]
        data_wavelengths = f.root.Wavelength[:]
        lerData = f.root._f_get_child('Mode_LER')[:]
    
    tree, a, b, t, sp, lat_tp, lon_tp, time_data = obtainTempPresProfile_fromMeteoData.assemble_tree(meteofile, meteofile_sp)
    
    dynamicScalingSwitch = True
    outputname = 'retrievalResults_NNmodel_dynamicScaling.h5'
    modelpath = os.path.join('./', 'models')
    modelpathAP = os.path.join('./', 'models', 'apriori')

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)        
        
    outputfilename = os.path.join(orbit_path, outputname)
    
    with tables.open_file(outputfilename,
              mode='a') as f:
        if '/nn_OE' not in f:
            group = f.create_group("/", 'nn_OE', 'output from NN alh algorithm')
            if '/nn_OE/output' not in f:
                f.create_table(group, 'output', NNoutput, 'results')
    
    # scl,pxls to process from orbit 858:

    with nc.Dataset(tropl2out, 'r') as f:
        alh_mid_height_o2a = f['PRODUCT']['aerosol_mid_pressure'][0,:,:]/100
    
    mask_o2a = np.ma.nonzero(np.ma.getmask(alh_mid_height_o2a) == 0)
    pixels_to_process = list(zip(mask_o2a[0], mask_o2a[1]))
    
    try:
        with tables.open_file(outputfilename,
                  mode='r') as f:    
            scl_done = f.root.nn_OE.output.cols.scl[:]
            pxl_done = f.root.nn_OE.output.cols.pxl[:]
        
        ret_done = list(zip(scl_done,pxl_done))
        
    except Exception as e:
        
        print(scl_done, pxl_done, 'exists')
        
    for i in ret_done:
        if i in pixels_to_process:
            pixels_to_process.remove(i)

    pixels_to_process = np.array(pixels_to_process)
    
    #######

    priori = [ 750, 2.0 ]

    prioricovariance = np.diag([500.0, 1.0])**2
    maxiter = 12
    
    models = ['I__o2a', 'K_alp', 'K_aot', 'K_as758__o2a', 'K_as770__o2a']
    modelsAP = ['ap__zaer', 'ap__aot']

    count = pixels_to_process.shape[0]
    gapriori = False                # guess a priori parameters
    
    # retrieve pixel by pixel:
    with nc.Dataset(filenameRad, 'r') as fname_rad, nc.Dataset(filenameIrr, 'r') as fname_irr:
    
        for selPxl in pixels_to_process:
            
            print('pixel number: ', count)
            sys.stdout.write("\033[F")
            count-=1
            
            with tables.open_file(outputfilename,
                      mode='a') as f:
        
                scanline = selPxl[0]
                pixel = selPxl[1]    
                output = f.root.nn_OE.output.row
    
                startTime = time.time()

                wvl_irr, irr, snr_irr, wvl_rad, rad, snr_rad, \
                geodata, dTime, latB, lonB = readTROPOMIradIrr.readTROPOMIradIrrData_nc(fname_irr,
                                                                                        fname_rad,
                                                                                     scanline,pixel,
                                                                                     6)
                print('time taken for reading input data:', time.time() - startTime, 'seconds')
                print('scanline, pixel:', scanline, pixel)
                
                
                sza,vza,saa,vaa,lat,lon = geodata
                wvl_rad_idx = (wvl_rad>=758.0) & (wvl_rad <= 770.0)
                wvl_irr_idx = (wvl_irr>=758.0) & (wvl_irr <= 770.0)
                mu = np.cos(np.deg2rad(vza))
                mu0 = np.cos(np.deg2rad(sza))
                geo = [sza, vza, saa, vaa]
                
                
                """ spline-interpolate solar irradiance to radiance wavelength grid """
                
                instrWavl = wvl_rad[wvl_rad_idx]
                irr_new = convolution__module.spline(wvl_irr[wvl_irr_idx], irr[wvl_irr_idx], instrWavl)
                snr_irr_new = convolution__module.spline(wvl_irr[wvl_irr_idx], snr_irr[wvl_irr_idx], instrWavl)
                sun_norm_rad = rad[wvl_rad_idx]/(irr_new)
                
                refl = (np.pi*rad[wvl_rad_idx])/(mu0*irr_new)
                
                snr_sun_norm_rad = 1/np.sqrt(snr_irr_new**(-2) + snr_rad[wvl_rad_idx]**(-2))
                noise = sun_norm_rad / snr_sun_norm_rad
                
                crlat = latB
                crlon = lonB
                clat = lat
                clon = lon

                startTime = time.time()
                salb758, salb772 = extractsurfacealbedo.extractsurfalbedo_lite(dataLat, dataLon, data_wavelengths, lerData, month,[758., 772],
                                                                          lat,lon)
                print('time taken for surf albedo interpolation:', time.time() - startTime, 'seconds')
                
                startTime = time.time()
                PTdata = obtainTempPresProfile_fromMeteoData.calculate_levels(tree, 
                                                        (lat,lon,dTime) , 
                                                        a, b, t, sp, lat_tp, lon_tp, 
                                                        time_data)
                print('time taken for meteo interpolation:', time.time() - startTime, 'seconds')
                
                interpolant_PT = interpolate.interp1d(PTdata[2], PTdata[3], axis=0, fill_value='extrapolate', bounds_error=False)
                psfc = PTdata[-1]
                asfc = [salb758, salb772]
                
                model =  aerosolLayerHeight(
                      prior = priori, 
                      priorCov=prioricovariance, 
                      guessApriori = gapriori,
                      dynamicScaling = dynamicScalingSwitch,
                      geo = geo,
                      models = models,
                      modelsAP = modelsAP,
                      otherModelParam=[asfc[0], asfc[-1]], 
                      parameterNames=['z_aer', 'aot', 'As_758','As_770' ],
                      fitparams=['z_aer', 'aot'],
                      verbose=1,
                      observation=refl,
                      observationError=noise,
                      independentVariable=instrWavl,
                      HRwav = t_wavel, 
                      gaussian_weights = weights,
                      mu = mu, 
                      mu0 = mu0, 
                      asfc = asfc[0], 
                      psfc = psfc,
                      temp = interpolant_PT, 
                      modelPath = modelpath,
                      modelPathAP = modelpathAP,
                      pxl=pixel
                      )
                
                if dynamicScalingSwitch:
                    model()
                
                startTime = time.time()
                oe = OE(model=model, maxiter=maxiter, stateVectorConvThreshold=1e0)
                palh = oe.model.initialStateVector[0]
                paot = oe.model.initialStateVector[1]
                
                if  palh < 0 or paot < 0.0:
                    
                    oe.model.initialStateVector = [650.0, 2.0]
                    
                oe()
                stoptime = time.time()
                oeTime = stoptime - startTime
                print('time taken for retrieval:', oeTime, 'seconds')
                print('pixels left: ', count)
                est_time = (stoptime - startTime) * count / 60.0
                print('estimated time: ', est_time , 'minutes' )
                
                wrmse = oe.answer['weighted root mean square difference']
                chi2=oe.answer['cost function']
                dof=oe.answer['degrees of freedom']
                aPosErrCovMat=oe.answer['a posteriori noise error covariance matrix']
                numiter=oe.answer['number of iterations']
                svConvCrit=oe.answer['state vector convergence criterium']
                ret_alh=oe.answer['a posteriori state vector'][0]
                ret_aot=oe.answer['a posteriori state vector'][1]
                converged=oe.answer['converged']
                apriori_alh=model.initialStateVector[0]
                apriori_aot=model.initialStateVector[1]
                
#%%
                # write output:
                output['crlat'] = crlat
                output['crlon'] = crlon
                output['clat'] = clat
                output['clon'] = clon
                output['scl'] = scanline
                output['pxl'] = pixel
                output['ret_alh'] = ret_alh
                output['ret_alh_km'] = oe.model.descale_hPa(ret_alh)
                output['ret_aot'] = ret_aot    
                output['apriori_alh'] = apriori_alh
                output['apriori_alh_km'] = oe.model.descale_hPa(apriori_alh)

                output['apriori_aot'] = apriori_aot
                
                output['wrmse'] = wrmse
                output['chi2'] = chi2
                output['dof'] = dof
                output['aPosErrCovMat'] = aPosErrCovMat
                output['numiter'] = numiter
                output['svConvCrit'] = svConvCrit
                output['converged'] = converged
                output['asfc'] = asfc[0]
                output['oeTime'] = oeTime
                
                output.append()                            
                
                print('retrieved alh:', oe.model.descale_hPa(ret_alh), 'km' )
                    
                print('\n ')                
                f.root.nn_OE.output.flush()
                model.close()
                
                if converged:
                
                    priori = [ret_alh, ret_aot]
                    gapriori = False
                
                else:
                    
                    gapriori = True

if __name__ == "__main__":
    
    main()
