# ==============================================================================
# Truncated SVD regression with confidence intervals from stationary bootstrap
# ==============================================================================
# Documentation on principal component analysis, cross validation, stationary
# bootstrap and choice of optimal block length:
# https://www.statsmodels.org/dev/generated/statsmodels.multivariate.pca.PCA.html
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#arch.bootstrap.optimal_block_length
# ==============================================================================


import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
import arch.bootstrap as abs
from tqdm import tqdm

class model:
    def __init__(self, x_in, y_in, w_in, block_length = None, nboot = 500, ncomp = None):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array)
        # block_length: average block length for stationary bootstrap, will be
        #               determined automatically if not set
        # nboot: number of bootstrap samples
        # ncomp: number of principal components to be retained, if not supplied,
        #        the number will be determined automatically using cross validation

        print ("Principal component regression")
        print ("Confidence intervals based on stationary bootstrap")

        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]

        xpcr = np.reshape (np.transpose (x_in[:,:,:-nb], axes=(1,0,2)), (nb*nt,nn-nb))
        # The array xpcr contains *only* the variables for interventions and not the
        # dummy variables for states
        xall = np.reshape (np.transpose (x_in[:,:,:], axes=(1,0,2)), (nb*nt,nn))
        ypcr = np.reshape (np.transpose (y_in[:,:]), nb*nt)
        wpcr = np.reshape (np.transpose (w_in[:,:]), nb*nt)   

        # ----------------------------------------------------------------------
        # Determine hyperparameters by cross validation if they have not been specified
        if ncomp is None:
            cv = TimeSeriesSplit (n_splits=5)
            cv_score = []
            cv_min = 1e99
            print ("Hyperparameter selection using cross-validation")
            print ('Detemining optimal number of components by cross-validation...')
            for ncomp in range (2,20):
                # PCA without centering
                pca = sm.PCA (xpcr, ncomp=ncomp,
                             standardize=False, demean=False, normalize=False)    
                xpca = np.zeros([nb*nt,ncomp+nb])
                # Reassemble array of exogenous variables
                xpca[:,:ncomp] = pca.factors
                xpca[:,ncomp:] = xall[:,nn-nb:]
                cv_score_tmp = 0
                for i in cv.split (xpca, ypcr):
                    model_pca = sm.WLS (ypcr[i[0]], xpca[i[0],:], weights = wpcr[i[0]]).fit()
                    ypred = model_pca.predict (xpca[i[1],:])
                    # Add (weighted) squared sum of residual to cross validation score
                    cv_score_tmp += np.average ((ypred-ypcr[i[1]]) ** 2 *wpcr[i[1]])
                    cv_score.append (cv_score_tmp)
                if cv_score_tmp < cv_min: # we have found a better model
                    cv_min = cv_score_tmp
                    ncomp_opt = ncomp
                print('%i compomnents: CV score = %f' % (ncomp, cv_score_tmp))

            print ('Optimal number of compponents: %i' % ncomp_opt)
            ncomp = ncomp_opt
        else:
            print ('Using %i principal components.' % ncomp)
            
        # ----------------------------------------------------------------------
        # Fit the model with the optimal number of principal components
        xpcr = np.reshape (x_in[:,:,:-nb],(nb*nt,nn-nb))
        xall = np.reshape (x_in[:,:,:],(nb*nt,nn))
        ypcr = np.reshape (y_in[:,:],nb*nt)
        wpcr = np.reshape (w_in[:,:],nb*nt)   
        
        # PCA without centering
        pca = sm.PCA(xpcr, ncomp=ncomp,
                     standardize=False, demean=False, normalize=False)    
        xpca = np.zeros([nb*nt,ncomp+nb])
        xpca[:,:ncomp] = pca.factors
        xpca[:,ncomp:] = xall[:,nn-nb:]
        model_pca = sm.WLS (ypcr, xpca, weights = wpcr).fit()
        self.result = model_pca

        # Regression parameters from factor loadings
        params = np.zeros(nn)
        params[-nb:] = model_pca.params[-nb:]
        params[:nn-nb] = np.dot (pca.loadings, model_pca.params[:ncomp])
        self.params = params

        # ----------------------------------------------------------------------
        # Determine parameters for the stationary bootstrap if required        
        if block_length == None:
          # Use in-built function from ARCH package on residuals in each state
          # and then take the average value.
          restest = np.reshape(model_pca.resid,(nb,nt))
          opt_len = np.zeros(nb)
          for i in range(nb):
            opt_len[i] =  abs.optimal_block_length(restest[i,:])['stationary']
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

        for data,inx in tqdm(sbs.bootstrap(nboot)):
            ii = data[0].astype(int)           
            # Boostrap residuals
            # Same resampling for all time series
            ytest = np.reshape (model_pca.fittedvalues, (nb,nt)) + \
                np.reshape (model_pca.resid, (nb,nt))[:,ii]
            ytest = np.reshape (ytest, nb*nt)
            model_test = sm.WLS (ytest, xpca, weights = wpcr).fit()
            params_test = np.zeros(nn)
            params_test[-nb:] = model_test.params[-nb:]
            params_test[:nn-nb] = np.dot (pca.loadings, model_test.params[:ncomp])
            dparam.append ([params_test-params])

        dparam = np.reshape (np.array(dparam),(nboot,nn))
        # 95% confidence intervals from standard deviation of bootstrap sample
        self.err = np.sqrt (np.average (dparam[:,:]**2, axis=0)) * 1.96
        
    def conf_int(self):
        return np.stack ([self.params-self.err, self.params+self.err],axis=1)
