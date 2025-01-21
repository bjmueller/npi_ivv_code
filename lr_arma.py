# ==============================================================================
# Linear regression with ARMA(p,q) errors
# ==============================================================================
# Relevant documentation:
# State space models in statsmodels:
# https://www.statsmodels.org/stable/statespace.html
# https://www.statsmodels.org/stable/statespace.html#models-and-estimation
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEModel.html#statsmodels.tsa.statespace.mlemodel.MLEModel
# Computation of autocorrelation function (ACF):
# https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html
# ==============================================================================


import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.statespace.tools as tools
from statsmodels.tsa.stattools import acf

class model(sm.tsa.statespace.MLEModel):

    # Regression model with ARMA (p,q) errors in state space formulation
    # (Harvey representation).
    
    def __init__(self, x_in, y_in, w_in, keys_in, p=1, q=0, \
                 params_in = None, phi_in = None, theta_in = None, sigma2_in = None):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # y_in: response variable (nb*nt array)
        # w_in: weights (nb*nt array) -- currently unused
        # keys_in: names of explanatory variables (nn array)
        # p, q: AR and MA order of errors
        # params_in: starting values for parameters, can be used for faster
        #            convergence, e.g., when scanning grids of parameters (nn array)
        # phi_in:    starting values for \Phi coefficients of AR terms
        # theta_in:  starting values for \Theta coefficients of MA terms
        # sigma2_in: startinv value for variance

        # Returned parameter estimates:
        # params[:nn-nb]: NPI regression coefficients
        # params[nn-nb:nn]: state fixed effects
        # params[nn:nn+p]: coefficients phi_1,...,phi_p of AR(p) terms
        # params[nn+p:nn+p+q]: coefficients theta_1,...,theta_q of MA(q) terms
        # params[-1]: variance (sigma^2) of innovations

        #-----------------------------------------------------------------------
        # Initialise parameters and arrays for response variable and explanatory
        # variables
        self.n_arima = p+q
        self.p = p
        self.q = q
        self.nt = np.shape(x_in)[1]
        self.nb = np.shape(x_in)[0]
        self.nn = np.shape(x_in)[2]
        self.x_in = x_in
        self.y_in = y_in
        self.w_in = w_in        
        self.params_in = params_in
        try: self.phi_in = np.reshape (phi_in,len(phi_in))
        except: self.phi_in = None
        try: self.theta_in = np.reshape (theta_in,len(theta_in))
        except: self.theta_in = None
        self.sigma2_in = sigma2_in
        self.sdim = max(p,q+1)
        self.keys_in = keys_in[0:self.nn]
        endog = np.reshape (y_in,self.nb*self.nt)
        exog = np.reshape (x_in[:,:,:self.nn],(self.nb*self.nt,self.nn))
        (self._k_exog, exog) = tools.prepare_exog(exog)

        self.k_exog = self._k_exog
        self.state_regression = False
        self.mle_regression = True
        self.measurement_error = False
        self.state_error=True

        #-----------------------------------------------------------------------
        # Initialise MLE model
        super(model, self).__init__(
            endog=endog, exog=exog, k_states=self.sdim, k_posdef=1,\
            initialization="known", initial_state = np.zeros (self.sdim),\
            initial_state_cov=1e-3*np.eye(self.sdim,self.sdim))
        k_states = self.k_states
        
        params = np.zeros(self.nn+1+self.n_arima)
        params[-1 ]= 4e-3 #starting value for sigma2
                
        # Initialise matrices and vectors in state space model -- will be
        # overwritten later in update() function, but shape is important:
        self.ssm["design"] = np.eye(1,self.sdim)
        self.ssm["obs_cov"] = np.array([[0.]])
        self.ssm['obs_intercept'] = np.zeros((1,self.nobs),dtype=np.complex128)
        self.ssm['obs_intercept'][0,:] = np.dot(self.exog,params[:self.nn])
        self.ssm["selection"] = np.eye(self.sdim,1)
        self.ssm["selection"][1:self.q+1,0] = params[self.nn+self.p:self.nn+self.n_arima]
        self.ssm["transition"] = np.tensordot (np.eye(self.sdim,self.sdim,k=1,dtype=np.complex128), np.ones(self.nobs),axes=0)
        if self.p>0:
            self.ssm["transition"][:self.p,0,:] = np.reshape (np.outer(params[self.nn:self.nn+self.p], np.ones(self.nt*self.nb)), (self.p,self.nobs))
        for ib in range(self.nb):
            # Restart uncorrelated noise time series for each state
            # by zeroing the transition matrix between the individual patches.
            # This needs to be checked carefully!
            self.ssm["transition"][:,:,ib*self.nt-1] = 0.0
            self.ssm['obs_intercept'][0,ib*self.nt] = self.y_in[ib,0]
        # Cut noise time series into five chunks to filter low frequencies whose power
        # cannot be adequately estimated
        self.ssm["transition"][:,:,::self.nt//5] = 0.0
        self.ssm["state_cov"] = params[-1]*np.array([[1.]])
        self.positive_parameters = np.array([-1])
        
    @property
    def param_names(self):
        keys_arma =self.keys_in[0:self.nn].copy()
        keys_arma.extend(["phi%i" % i for i in range(1,self.p+1)])
        keys_arma.extend(["theta%i" % i for i in range(1,self.q+1)])
        keys_arma.extend(["sigma2"])
        return keys_arma
    
    @property
    def start_params(self):
        """
        Initial values
        """
        params = np.zeros(self.nn+1+self.n_arima)
        # To obtain reasonable starting values for the parameters, perform
        # OLS regrssion
        model_OLS = sm.OLS (np.reshape (self.y_in,[self.nb*self.nt]), np.reshape(self.x_in[:,:,:self.nn], [self.nb*self.nt,self.nn]))
        res_ols = model_OLS.fit()
        params[:self.nn] = res_ols.params[:]
        params[-1] = np.average (res_ols.resid**2)

        #If starting parameters (e.g., from a similar or unconverged model) are specified, use these:
        if self.params_in is not None:
            params[:self.nn] = self.params_in[:self.nn]
        if self.phi_in is not None:
            p = min (len(self.phi_in), self.p)
            params[self.nn:self.nn+p] = self.phi_in[:p]
        if self.theta_in is not None:
            q = min (len(self.theta_in), self.q)
            params[self.nn+self.p:self.nn+self.p+q] = self.theta_in[:q]
        if self.sigma2_in is not None:
            params[-1] = self.sigma2_in
            
        return params

    def transform_params(self, unconstrained):
        """
        If you need to restrict parameters
        For example, variances should be > 0
        Parameters maybe have to be within -1 and 1
        """
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = (
            constrained[self.positive_parameters] ** 2)
    
        return constrained

    def untransform_params(self, constrained):
        """
        Need to reverse what you did in transform_params()
        """
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = (
            unconstrained[self.positive_parameters] ** 0.5
        )
        return unconstrained

    def update(self, params, **kwargs):
        params = super(model, self).update(params, **kwargs)
        self.ssm['obs_intercept'][0,:] = np.dot(self.exog,params[:self.nn])

        if self.p>0: # Update AR coefficients in the transition matrix
            self.ssm["transition"][:self.p,0,:] = np.reshape(np.outer(params[self.nn:self.nn+self.p],np.ones(self.nt*self.nb)),(self.p,self.nobs))
        for ib in range(self.nb):
            # Restart uncorrelated noise time series for each state
            # by zeroing the transition matrix between the individual patches.
            # This needs to be checked carefully!
            self.ssm["transition"][:,:,ib*self.nt-1]=0.0
            # Fix the intercept of the observation so that the first data point
            # fits the observed R(t) in each state (and thus exclude it
            # for effect estimation).
            self.ssm['obs_intercept'][0,ib*self.nt] = self.y_in[ib,0]

        # Update MA coefficients in the selection matrix
        self.ssm["selection"][1:self.q+1,0] = params[self.nn+self.p:self.nn+self.n_arima]
        # Update \sigma^2 in the state covariance matrix
        self.ssm["state_cov"] = params[-1] * np.array([[1.]])
        # Cut noise time series into five chunks to filter low frequencies whose power
        # cannot be adequately estimated
        self.ssm["transition"][:,:,::self.nt//5] = 0.0
        

def arma_grid (x, y, wt, keys_x, p_max=3, q_max=6, maxiter=120):

    # Creates a grid of regression models with ARMA (p,q) errors,
    # e.g., for lag selection or comparison of regression parameters
    # and error bars.
    
    # Arguments:
    # x: nb*nt*nn array of explanatory variables where
    #    nb = number of geographical entities
    #    nt = number of time step
    #    nn = number of NPI variables + number of entities
    # y: response variable (nb*nt array)
    # w: weights (nb*nt array) -- currently unused
    # keys_x: names of explanatory variables (nn array)
    # p_max, q_max: Maximum AR and MA orders in grid

    arma_grid = [ [None for p in range(p_max+1)] for q in range(q_max+1) ]

    nt = np.shape(x)[1]
    nb = np.shape(x)[0]
    nn = np.shape(x)[2]

    # We want to find the model with the smallest BIC -- initialise with large value
    bic = 1e30
    
    # Loop over p and q, recyclying parameter values from lower-order models as
    # initial guess values in the iteration
    for q in range(q_max+1):
        for p in range (p_max+1):
            print('\n---------------------------------------------------------------------')
            print('Fitting model with ARMA(%i,%i) errors.' % (p,q))
            arma_grid[q][p] = model (x,y,wt,keys_x,p=p,q=q).fit(maxiter=maxiter)

            print ("BIC:", arma_grid[q][p].bic)
            if arma_grid[q][p].bic < bic:
                bic = arma_grid[q][p].bic
                optim_p = p
                optim_q = q

        print ("Optimal model: ARMA (%i,%i)." % (optim_p, optim_q))
                
    return arma_grid
