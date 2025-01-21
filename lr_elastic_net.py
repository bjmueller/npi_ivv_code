# ==============================================================================
# Elastic net regression with confidence intervals from stationary bootstrap
# ==============================================================================
# Documentation on elastic net regression, cross validation, stationary
# bootstrap and choice of optimal block length:
# https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.ElasticNetCV.html
# https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#arch.bootstrap.optimal_block_length
# ==============================================================================


import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
import arch.bootstrap as abs
from tqdm import tqdm

class model:
    def __init__(self, x_in, y_in, w_in, block_length = None, nboot = 200, alpha = None, l1_ratio = None):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array)
        # block_length: average block length for stationary bootstrap, will be
        #               determined automatically if not set
        # nboot: number of bootstrap samples
        # alpha: sum of L2 (Tikhonv) and L1 (Lasso) coefficients
        # l1_ratio: ratio \Theta / (\Theta + \Gamma) for weighting of L1 and L2 terms

        print ("Elastic net regression")
        print ("Hyperparameter selection using cross validation")
        print ("Confidence intervals based on stationary bootstrap")

        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]

        # Only include intervention variables, not the dummy variables for states
        x_cv = np.reshape (np.transpose (x_in[:,:,:-nb], axes=(1,0,2)), (nb*nt,nn-nb))
        y_cv = np.reshape (np.transpose (y_in[:,:]), nb*nt)
        w_cv = np.reshape (np.transpose (w_in[:,:]), nb*nt)

        # Determine hyperparameters by cross validation if they have not been specified
        if alpha is None:
            alphas = 10.0 ** np.arange(-4, -1.4, 0.125)
        else:
            alphas = [alpha]
        if l1_ratio is None:
            l1_ratios = [1e-3, 2e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        else:
            l1_ratios = [l1_ratio]
        
        mod_cv = ElasticNetCV (alphas=alphas, l1_ratio = l1_ratios,
                               copy_X=True, fit_intercept=True,
                               max_iter=3000, cv=TimeSeriesSplit(n_splits=5))
        mod_cv.fit (x_cv, y_cv, sample_weight = w_cv)

        print ("Optimal values from cross-validation:")
        print ("alpha = %f" % mod_cv.alpha_)
        print ("l1_ratio = %f" % mod_cv.l1_ratio_)

        # Now fit the full model
        x_en = np.reshape (x_in[:,:,:-nb], (nb*nt,nn-nb))
        y_en = np.reshape (y_in[:,:], nb*nt)
        w_en = np.reshape (w_in[:,:], nb*nt)

        mod_en = ElasticNet (alpha=mod_cv.alpha_, l1_ratio=mod_cv.l1_ratio_,
                             copy_X=True,fit_intercept=True, max_iter=3000)
        mod_en.fit (x_en, y_en, sample_weight=w_en)
        self.result = mod_en

        self.params = np.zeros(nn)
        self.params[:nn-nb] = mod_en.coef_ # regression parameters
        # Fixed effects for states all get the same value (common intercept)
        self.params[-nb:] = mod_en.intercept_
        ypred = mod_en.predict(x_en)
        resid = np.reshape(y_en - ypred, (nb,nt))
        
# ------------------------------------------------------------------------------
        # Determine parameters for the stationary bootstrap if required

        if block_length == None:
          # Determine optimal block length for stationary bootstrap.
          # Use in-built function from ARCH package on residuals in each state
          # and then take the average value.
          #print(abs.optimal_block_length(restest[i,:]))
          opt_len = np.zeros(nb)
          for i in range(nb):
            opt_len[i] =  abs.optimal_block_length (resid[i,:])['stationary']
          block_length = np.round (np.average (opt_len))
          print ("Average block length not specified, using length of %i." % block_length)
        else:
          print ("Using specified block length of %i." % block_length)

# ------------------------------------------------------------------------------
        # Perform the bootstrap
        sbs = abs.StationaryBootstrap(block_length,np.arange(nt))

        print ("Boostrapping, resampling %i times." % nboot)
        dparam = [] # will contain the deviations of the regrssion parameters
                    # in the bootstrap sample from the point estimates
        for data,inx in tqdm (sbs.bootstrap (nboot)):
            ii = data[0].astype(int)           
            # Boostrap residuals
            # Same resampling for all time series
            yy = ypred + np.reshape(resid[:,ii], nb*nt)
            mod_bt = ElasticNet (alpha=mod_cv.alpha_, l1_ratio=mod_cv.l1_ratio_,
                                 copy_X=True,fit_intercept=True, max_iter=3000)
            mod_bt.fit (x_en, yy, sample_weight=w_en)
            dparam.append ([mod_bt.coef_ - self.params[:-nb]])
        dparam = np.reshape (np.array (dparam), (nboot,nn-nb))
        self.err = np.zeros(nn)
        # 95% confidence intervals from standard deviation of bootstrap sample
        self.err[:-nb] = np.sqrt (np.average (dparam[:,:]**2, axis=0)) * 1.96
    
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack ([self.params-self.err, self.params+self.err],axis=1)
