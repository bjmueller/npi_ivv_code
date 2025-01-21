# ==============================================================================
# Linear regression with two-way fixed effects, CIs from stationary bootstrap
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
                 asynchronous = True, resample_states = False):

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
        
        
        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]

        # Arrays of regressors include fixed effects in time
        xaux = np.zeros((nb,nt,nn+nt))
        xaux[:,:,:nn] = x_in[:,:,:]
        # Zero seasonal variables - will be taken care of later
        xaux[:,:,nn-nb-1]=0.0
        xaux[:,:,nn-nb-2]=0.0
        xaux[:,:,nn-nb-4]=0.0
        # Dummy variables for fixed effects
        for i in range (nt):
            xaux[:,i,nn+i] = 1.0
            
        res0 = sm.WLS (np.reshape(y_in,[nb*nt]), np.reshape(xaux,[nb*nt,nn+nt]),\
                   weights=np.reshape(w_in,[nb*nt])).fit()
        self.result = res0
        # In principle, residuals need to include time fixed effects in this case,
        # as they contain unmodelled noise. As long as we use the same resampling
        # for each state, this does not matter, however.
        #if asynchronous:
        #    resid = res0.resid + np.dot (res0.model.exog[:,nn:], res0.params[nn:])
        #    fittedvalues = np.dot (res0.model.exog[:,:nn], res0.params[:nn])
        #else:
        resid = res0.resid
        fittedvalues = res0.fittedvalues

        self.params = res0.params[:nn]
        # Regress fixed effects in terms of cos and and sin terms (hierarchical model)
        xsec = np.zeros((nb,nt,4))
        xsec[:,:, 0] = x_in[:,:,nn-nb-4] 
        xsec[:,:, 1] = 1.0
        xsec[:,:,-2] = x_in[:,:,nn-nb-2] 
        xsec[:,:,-1] = x_in[:,:,nn-nb-1]
        fe_time = np.reshape(np.dot(res0.model.exog[:,nn:], res0.params[nn:]),(nb,nt))
        lrs = sm.WLS (np.reshape(fe_time,nb*nt), np.reshape(xsec[:,:,:],(nb*nt,4)), \
                     weights = np.reshape(w_in,(nb*nt))).fit()
        self.params[nn-nb-4] = lrs.params[ 0]
        self.params[nn-nb-2] = lrs.params[-2]
        self.params[nn-nb-1] = lrs.params[-1]
        
        # ----------------------------------------------------------------------
        # Confidence intervals from bootstrap
        dparam = [] # will contain the deviations of the regrssion parameters
                    # in the bootstrap sample from the point estimates
        params_boot = [] # will contain the regression parameters in the
                         # boostrap sample                         
            
        # ----------------------------------------------------------------------
        # Determine parameters for the stationary bootstrap if required
        if block_length == None:
          # Use in-built function from ARCH package on residuals in each state                            
          # and then take the average value.                                                              
          restest = np.reshape (res0.resid,(nb,nt))
          #print(abs.optimal_block_length(restest[i,:]))                                                  
          opt_len = np.zeros(nb)
          for i in range(nb):
            opt_len[i] =  abs.optimal_block_length (restest[i,:])['stationary']
          block_length = np.round (np.average (opt_len))
          print ("Average block length not specified, using length of %i." % block_length)
        else:
          print ("Using specified block length of %i." % block_length)

# ------------------------------------------------------------------------------
        # Perform the bootstrap
        sbs = abs.StationaryBootstrap(block_length,np.arange(nt)) # temporal resampling
        ii = [data[0].astype(int) for data,inx in sbs.bootstrap(nboot*(nb+1))]

        sbsb = abs.IIDBootstrap(np.arange(nb)) # for resamling states
        iib = [data[0].astype(int) for data,inx in sbsb.bootstrap(nboot)]

        for iboot in tqdm(range(nboot)):
            yy = np.copy (np.reshape(res0.fittedvalues,(nb,nt)))
            # Boostrap residuals:
            if resample_states:
                tmp = np.copy(np.reshape(resid,(nb,nt))[iib[iboot],:])
            else:
                tmp = np.copy(np.reshape(resid,(nb,nt))[:,:])
            if asynchronous:    
                for ib in range(nb):
                    yy[ib,:] = np.reshape(fittedvalues,(nb,nt))[ib,:] + tmp[ib,ii[iboot*nb+ib]]
            else:
                yy = np.reshape(fittedvalues,(nb,nt)) + np.reshape(resid,(nb,nt))[:,ii[iboot*nb]]

            xx = xaux[:,:,:]
            ww = w_in[:,:]
            res_tmp = sm.WLS (np.reshape(yy,[nb*nt]), np.reshape(xx,[nb*nt,nn+nt]),\
                              weights=np.reshape(ww,[nb*nt])).fit()

            # Seasonal terms
            fe_time = np.reshape (np.dot(res0.model.exog[:,nn:], res0.params[nn:]), (nb,nt))
            lrs_tmp = sm.WLS (np.reshape(fe_time[:,ii[iboot*nb+nb]],nb*nt), np.reshape(xsec[:,ii[iboot*nb+nb],],(nb*nt,4)), \
                         weights=np.reshape(w_in[:,ii[iboot*nb+nb]], (nb*nt))).fit()
            res_tmp.params[nn-nb-4] = lrs_tmp.params[ 0]
            res_tmp.params[nn-nb-2] = lrs_tmp.params[-2]
            res_tmp.params[nn-nb-1] = lrs_tmp.params[-1]

            dparam.append ([res_tmp.params[:nn]-res0.params[:nn]])
            params_boot.append ([res_tmp.params[:nn]-res0.params[:nn]])

        dparam = np.reshape (np.array(dparam), (nboot,nn))
        params_boot = np.reshape (np.array(params_boot), (nboot,nn))
        
        self.dparams = dparam # Store this in case we need the actual distribution                        
        self.params_boot = params_boot
        self.params = res0.params[:nn]
        # 95% confidence interval from standard deviation of bootstrap sample
        self.err = np.sqrt (np.average(dparam[:,:]**2,axis=0)) * 1.96
        
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack ([self.params-self.err,self.params+self.err],axis=1)
