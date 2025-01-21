# ==============================================================================
# Linear regression with confidence intervals from frequency-domain power
# estimation of noise (based on Ebisuzaki's method)
# ==============================================================================
# Relevant Python documentation on WLS regression and Fast Fourier transform:
# https://www.statsmodels.org/dev/regression.html#model-classes
# https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
# ==============================================================================


import numpy as np
import statsmodels.api as sm

class model:
    def __init__(self, x_in, y_in, w_in, weighted_psd = False):
        
        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array)

        print ("WLS regression, confidence intervals based on Ebisuzaki's method.")

        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]

        res0 = sm.WLS (np.reshape(y_in,[nb*nt]), np.reshape(x_in,[nb*nt,nn]),\
                              weights = np.reshape(w_in, [nb*nt])).fit()
        self.result = res0
        resid = np.reshape (res0.resid, [nb,nt])
        self.params = res0.params
        nsig = 1.96
        wt = np.reshape (w_in, [16,nt])
        
        # Matrix and residuals for Fourier transform
        if weighted_psd:
            # In this case, we transform seudoinverse of the design matrix
            # and the weighted residuals.
            pinv_wexog = np.reshape (res0.model.pinv_wexog[:,:], [nn,nb,nt])          
            
            reswgt2 = np.concatenate ((resid*np.sqrt(wt), np.flip(resid*np.sqrt(wt),axis=1)), axis=1)
        else:
            # If we want to estimate the PSD of the unweighted residuals, we first need to multiply                         
            # the pseudoinverse by the square root of the weight matrix from the right.                              
            pinv_wexog = res0.model.pinv_wexog[:,:]
            pinv_wexog = np.matmul (pinv_wexog, np.diag (np.sqrt (np.reshape(w_in, [nb*nt]))))
            pinv_wexog = np.reshape (pinv_wexog, [nn,nb,nt])
            reswgt2 = np.concatenate ((resid, np.flip(resid,axis=1)), axis=1) # reflection padding for residuals

        # Fourier transform pseudoinverse of design matrix with reflection padding
        pinv_wexog2 = np.concatenate ((pinv_wexog, np.flip(pinv_wexog, axis=2)), axis=2) # reflection padding
        # Use unitary transform (option norm='orho')
        p_fft2 = np.abs (np.fft.fft (pinv_wexog2, axis=2, norm='ortho'))**2
        # Very simple form of downsampling for spectrum estimation original frequency grid                                   
        # (pairwise mean of frequency bins to obtain correct power spectral density on original grid)
        p_fft2 = 0.5*np.sum (np.reshape (p_fft2, [nn,nb,nt,2]), axis=3)

        # Now Fourier transform the residuals with reflection padding
        pwr_res2 = np.abs (np.fft.fft (reswgt2,norm='ortho'))**2
        pwr_res2 = 0.5 * np.sum (np.reshape (pwr_res2, [nb,nt,2]), axis=2)
        # 1.96-sigma confidence intervals (95% confidence level)
        err_corr = nsig * np.sqrt (np.sum (pwr_res2[np.newaxis,:]*p_fft2, axis=(1,2)))          
        self.err = err_corr

        
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack ([self.params-self.err, self.params+self.err], axis=1)
