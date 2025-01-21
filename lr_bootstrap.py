# ==============================================================================
# Linear regression with confidence intervals from stationary bootstrap
# ==============================================================================
# Relevant documentation on stationary bootstrap and optimal block length:
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#arch.bootstrap.optimal_block_length
# ==============================================================================

import numpy as np
import statsmodels.api as sm
import arch.bootstrap as abs
from tqdm import tqdm

class model:
    def __init__(self, x_in, y_in, w_in, block_length = None, nboot = 500, \
                 asynchronous = False, resample_states = False):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array)
        # block_length: average block length for stationary bootstrap, will be
        #               determined automatically if not set
        # nboot: number of bootstrap samples
        # asynchronous: asynchronous resampling in bootstrap for different states
        # resample_states: resample residuals across states in bootstrap

        print ("WLS regression, confidence intervals based on stationary bootstrap.")

        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]

        res0 = sm.WLS (np.reshape(y_in,[nb*nt]), np.reshape(x_in,[nb*nt,nn]),\
                              weights=np.reshape(w_in,[nb*nt])).fit()
        self.result = res0

# ------------------------------------------------------------------------------
        # Determine parameters for the stationary bootstrap if required
        if block_length == None:
          # Use in-built function from ARCH package on residuals in each state
          # and then take the average value.
          restest = np.reshape(res0.resid,(nb,nt)) # Array of residuals
          opt_len = np.zeros(nb)
          for i in range(nb): # get optimal  block length for each state
            opt_len[i] =  abs.optimal_block_length(restest[i,:])['stationary']
          block_length = np.round (np.average (opt_len))
          print ("Average block length not specified, using length of %i." % block_length)
        else:
          print ("Using specified block length of %i." % block_length)


# ------------------------------------------------------------------------------
        # Perform the bootstrap
        sbs = abs.StationaryBootstrap(block_length,np.arange(nt))
        ii = [data[0].astype(int) for data,inx in sbs.bootstrap(nboot*nb)]

        sbsb = abs.IIDBootstrap(np.arange(nb))
        iib = [data[0].astype(int) for data,inx in sbsb.bootstrap(nboot)]

        print ("Boostrapping, resampling %i times." % nboot)

        dparam = []
        params_boot = []
        
        for iboot in tqdm(range(nboot)):
            yy = np.copy(np.reshape(res0.fittedvalues,(nb,nt)))
            # Boostrap residuals
            if resample_states:
                tmp = np.copy(np.reshape(res0.resid,(nb,nt))[iib[iboot],:])
            else:
                tmp = np.copy(np.reshape(res0.resid,(nb,nt))[:,:])
            if asynchronous:
                for ib in range(nb):
                    yy[ib,:] = yy[ib,:] + tmp[ib,ii[iboot*nb+ib]]
            else:
                for ib in range(nb):
                    yy[ib,:] = yy[ib,:] + tmp[ib,ii[iboot*nb]]
                
            xx = x_in[:,:,:]
            ww = w_in[:,:]
            res_tmp = sm.WLS (np.reshape(yy,(nb*nt)), np.reshape(xx,[nb*nt,nn]),\
                              weights=np.reshape(ww,[nb*nt])).fit()
            dparam.append([res_tmp.params-res0.params])
            params_boot.append([res_tmp.params-res0.params])

        dparam = np.reshape(np.array(dparam),(nboot,nn))
        params_boot = np.reshape(np.array(params_boot),(nboot,nn))

        self.dparams = dparam # Store this in case we need the actual distribution
        self.params_boot = params_boot # Store parameter values as well
        self.params = res0.params
        # 95% confidence interval from standard deviation of bootstrap sample
        self.err = np.sqrt (np.average (dparam[:,:]**2, axis=0)) * 1.96
        
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack ([self.params-self.err, self.params+self.err],axis=1)
