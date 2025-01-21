# ==============================================================================
# Random forest regression with confidence intervals from stationary bootstrap
# ==============================================================================
# Documentation on cross validation, random forest regression, stationary
# bootstrap and choice of optimal block length:
# https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#arch.bootstrap.optimal_block_length
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import arch.bootstrap as abs
from tqdm import tqdm

class rf_results:
    def __init__ (self, x_in, y_in, wt_in, max_depth = None, n_estimators = None, \
                  max_features = None, nboot = 500, block_length = 20, ref_state = 1):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array)
        # max_depth: maximum depth of trees, will be determined by cross-validation
        #            if not specified
        # n_estimators: number of trees, will be determined by cross-validation
        #            if not specified
        # max_features: maximum features to be considered at each split
        # nboot: number of bootstrap samples
        # block_length: average block length for stationary bootstrap, will be
        #               determined automatically if not set
        # ref_state: calculate effect size with respect to no-NPI reference state
        #            (ref_state=0) or actual NPI activation profile (ref_state=1)

        nt = np.size(x_in[0,:,0])
        nb = np.size(x_in[:,0,0])
        nn = np.size(x_in[0,0,:])
        wt = np.reshape(wt_in, nb*nt)
        self.ref_state = ref_state

        # ----------------------------------------------------------------------
        # Determine optimal hyperparameters by cross validation, unless
        # they are already specified
        if max_depth is None:
            max_depth_test = [i for i in range(2,20)]
        else:
            max_depth_test = [max_depth]
        if n_estimators is None:
            n_estimators_test = [50, 71, 100, 141, 200]
        else:
            n_estimators_test = [n_estimators]
        if max_features is None:
            max_features_test = [1, 2, 4, 8, 16]
        else:
            max_features_test = [4]
            
        param_grid = {
            'bootstrap': [True],
            'max_depth': max_depth_test,
            'n_estimators': n_estimators_test,
            'max_features': max_features_test}

        # For CV using a time series split, the time series for the
        # states need to be interleaved, so transpose array accordingly:
        xrf = np.reshape (np.transpose (x_in[:,:,:], axes = (1,0,2)), (nb*nt,nn))
        yrf = np.reshape (np.transpose (y_in[:,:]), nb*nt)

        print ('Cross validation for hyperparameter selection...')
        grid_search = GridSearchCV (estimator = RandomForestRegressor (random_state = 0),
                                    param_grid = param_grid, cv = TimeSeriesSplit (n_splits=5), n_jobs = -1)
        grid_search.fit (xrf, yrf)

        max_depth    = grid_search.best_params_['max_depth']
        n_estimators = grid_search.best_params_['n_estimators']
        max_features = grid_search.best_params_['max_features']
        print ('Running with')
        print ('max_depth    = %i' % max_depth) 
        print ('n_estimators = %i' % n_estimators) 
        print ('max_features = %i' % max_features) 
       
        # ----------------------------------------------------------------------
        # Now perform regression with the optimal hyperparameters
        print ('Fitting main model...')
        xrf = np.reshape (x_in, (nt*nb,nn))
        yrf = np.reshape (y_in, (nt*nb))
        self.model = RandomForestRegressor (bootstrap=True,random_state=0, max_depth=max_depth,
                                           n_estimators=n_estimators, max_features=max_features)
        self.model.fit (xrf, yrf)
        self.predict = np.reshape (self.model.predict(xrf), (nb,nt))
        self.resid = self.predict - y_in

        # Extract linear effect sizes
        p0  = np.zeros(nn)
        eff = np.zeros(nn)
        for itest in range(nn):
            p0 = xrf.copy()
            if ref_state == 0:
                p0[:,:nn-2] = 0.
            elif ref_state != 1:
                raise ValueError ("Argument ref_state must be 0 or 1.")
            p0[:,itest] = 0.
            p1 = xrf.copy()
            if ref_state == 0:
                p1[:,:nn-2] = 0.
            p1[:,itest] = 1.
            eff[itest] = np.average (self.model.predict(p1) - self.model.predict(p0), weights = wt)
        self.params = eff

        # ----------------------------------------------------------------------
        # Error bars for parameters from stationary bootstrap
        # With random forest regression, there is a risk of overfitting,
        # so, by default, we bootstrap cases instead of residuals.

        print ('Error bars from stationary bootstrap...')
        if block_length == None:
            # Determine optimal block length for stationary bootstrap.
            # Use in-built function from ARCH package on R(t) data for
            # each state and then take the average value.
            opt_len = np.zeros(nb)
            for i in range(nb):
                opt_len[i] = abs.optimal_block_length (y_in[i,:])['stationary']
                block_length = np.round (np.average (opt_len))
                print ("Average block length not specified, using length of %i." % block_length)
            else:
                print ("Using specified block length of %i." % block_length)
          
        sbs = abs.StationaryBootstrap (block_length, np.arange(nt))
        dparam = []
        # Each random forest in the bootstrap ought to be intialised with a
        # different random state -- use irnd as a counter for this purpose.
        irnd = 0
        for data,inx in tqdm (sbs.bootstrap(nboot)):
            ii=data[0].astype(int)
            irnd += 1
            xrf = np.reshape (x_in[:,ii,:], (nt*nb,nn))
            yrf = np.reshape (y_in[:,ii], nt*nb)
            regr = RandomForestRegressor (bootstrap = True, max_depth = max_depth, random_state = irnd, \
                                          n_estimators = n_estimators, max_features = max_features)
            regr.fit (xrf, yrf)
             
            p0 = np.zeros(nn)
            eff = np.zeros(nn)
            for itest in range(nn):
                p0 = xrf.copy()
                if ref_state == 0:
                    p0[:,:nn-2] = 0.              
                p0[:,itest] = 0.
                p1 = xrf.copy()
                if ref_state == 0:
                    p1[:,:nn-2] = 0.
                p1[:,itest] = 1.
                eff[itest] = np.average (regr.predict(p1) - regr.predict(p0), weights = wt)
            dparam.append ([eff-self.params])
        dparam = np.reshape (np.array (dparam), (nboot,nn))
        self.err = np.sqrt (np.average (dparam[:,:]**2, axis = 0)) * 1.96
  
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack ([self.params-self.err, self.params+self.err], axis=1)

    def plot_importance(self, keys_npi):
        # Plot feature importances (Gini importances)
        nn = len (self.params)
        feat_imp = self.model.feature_importances_
        
        plt.figure(figsize=(6.4,18))
        plt.errorbar(feat_imp,-np.arange(nn),xerr=[feat_imp,feat_imp*0],fmt='o')
        plt.ylabel('NPI type')
        plt.yticks(-np.arange(nn),labels=keys_npi[:nn],rotation=0,minor=False)
        plt.ylim(-nn+0.5,0.5)
        plt.xlabel('Feature importance')
        plt.xlim(0,0.16)
        plt.axvline(10,linestyle='dotted')
        plt.minorticks_on()
        plt.savefig('feature_importance.pdf')
