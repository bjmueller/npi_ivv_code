# ==============================================================================
# Dynamical model (renewal equation) with CIs from stationary bootstrap
# ==============================================================================
# Documentation on state space models and bootstrap:
# https://www.statsmodels.org/stable/statespace.html
# https://www.statsmodels.org/stable/statespace.html#models-and-estimation
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEModel.html#statsmodels.tsa.statespace.mlemodel.MLEModel
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap
# https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html#arch.bootstrap.optimal_block_length
# ==============================================================================

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels.multivariate.pca as pca
import statsmodels.tsa.statespace.tools as tools
import arch.bootstrap as abs
from tqdm import tqdm

dtype = np.complex128

# Class for fitting an individual model:
class renewal(sm.tsa.statespace.MLEModel):
    
    def __init__(self, x_in, case_in, y_in, w_in, delta_r_var, keys_in, params_in = None, \
                 cut = 3, fix_vac = False):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # case_in: case data to be fitted (nb*nt array)
        # y_in: R(t) as response variable for determining initial guess values
        #            for effects using WLS (nb*nt array)
        # w_in: weights: weights for WLS (nb*nt array)
        # delta_r_var: change of ln R(t) due to variants relative to wild type (nb*nt array)
        # keys_in: names of explanatory variables (nn array)
        # params_in: starting values for parameters, can be used for faster
        #            convergence
        # cut: number of data points to be cut from the time series, has
        # to be 3 for current version of this module, but is already included
        # to accommodate renewal models with a different infectivity function
        # fix_vac: use corrected treatment of vaccine effect if true
        
        # Returned parameter estimates:
        # params[:nn-nb]: NPI regression coefficients
        # params[nn-nb:nn]: state fixed effects
        # params[nn]: scaling factor variance

        #-----------------------------------------------------------------------
        # Initialise parameters and arrays for response variable and explanatory
        # variables

        self.fix_vac = fix_vac
        self.cut = cut
        self.nt = np.shape(x_in)[1]-self.cut
        self.nb = np.shape(x_in)[0]
        self.nn = np.shape(x_in)[2]
        self.keys_x = keys_in
        
        self.x_in = np.copy(x_in[:,self.cut:,:])
        if self.fix_vac:
            # Transform -log_2 (1-V) back to the fraction V of vaccinated individuals:
            self.x_in [:,:,self.nn-self.nb-3] = 1 - 2 ** (-self.x_in [:,:,self.nn-self.nb-3])
        self.y_in = np.copy(y_in[:,self.cut:])
        self.w_in = np.copy(w_in[:,self.cut:])
        self.delta_r_var = np.copy(delta_r_var[:,self.cut:])
        # In the statespace model, we need the NPI variables for the NEXT
        # time step in the transition matrix, so shift to left by one element.
        # The last time step can stay -- not needed for estimation.
        self.x_in[:,:-1,:] = self.x_in[:,1:,:]
        self.y_in[:,:-1] = self.y_in[:,1:]
        self.w_in[:,:-1] = self.w_in[:,1:]
        self.delta_r_var[:,:-1] = self.delta_r_var[:,1:]
        
        self.case_in = np.copy(case_in[:,self.cut:])
        self.initial_cases = np.copy(case_in[:,:self.cut+1])

        self.delta_r_var = np.reshape(self.delta_r_var,self.nb*self.nt) 
        
        self.params_in = params_in

        endog = np.reshape(self.case_in,self.nb*self.nt)
        exog = np.reshape(self.x_in,(self.nb*self.nt,self.nn))
        (self._k_exog, exog) = tools.prepare_exog(exog)
        self.k_exog = self._k_exog
        self.state_regression = False
        self.mle_regression = True
        self.measurement_error = False
        self.state_error=True

        #-----------------------------------------------------------------------
        # Initialise MLE model
        super(renewal, self).__init__(
            endog=endog, exog=exog, k_states=self.cut+1, k_posdef=1,\
            initialization="known", initial_state = np.flip(case_in[0,0:self.cut+1]),\
            initial_state_cov=1e-3*np.eye(self.cut+1,self.cut+1))

        k_states = self.k_states
        nobs = self.nobs
        
        params = np.zeros(self.nn+1)
        if self.params_in is not None:
            params[:] = self.params_in[:]
        else:
            params[-1]=1.
                
        # Initialise matrices and vectors in state space model -- will be
        # overwritten later in update() function, but shape is important:
        self.ssm["design"] = np.eye (1,self.k_states)
        self.ssm["obs_cov"] = np.array ([[0.]])
        self.ssm['obs_intercept'] = np.zeros ((1,self.nobs), dtype=dtype)
        self.ssm['state_intercept'] = np.zeros ((4,self.nobs), dtype=dtype)

        self.ssm["selection"] = np.array ([[1.],[0.],[0.],[0.]])
        self.ssm["transition"] = np.zeros ((4, 4, self.nobs), dtype=dtype)
        self.ssm["obs_cov"] = np.zeros ((1, 1, self.nobs))
        self.ssm["state_cov"] = np.zeros ((1, 1, self.nobs))
           
        self.positive_parameters = np.array([-1])
        
    @property
    def param_names(self):
        keys_dyn=self.keys_x[0:self.nn].copy()
        keys_dyn.extend(['sigma2'])
        return keys_dyn
    
    @property
    def start_params(self):
        """
        Initial values
        """
        params = np.zeros(self.nn+1)
        if self.params_in is not None:
            # Use initial guess values provided by user
            params[:] = self.params_in[:]
        else:
            params[-1] = 1.
            # Initial guess values from WLS
            model_wls = sm.WLS (np.reshape (self.y_in, [self.nb*self.nt]), \
                                np.reshape (self.x_in[:,:,:self.nn], [self.nb*self.nt,self.nn]), \
                                weights=np.reshape (self.w_in, [self.nb*self.nt]))
            res_wls = model_wls.fit()
            params[:self.nn] = res_wls.params[:]
            if self.fix_vac: # Manually set initial guess value for vaccine efficiency
                params[self.nn-self.nb-3] = 0.7 

            
        self.ssm["design"] = np.eye(1,self.k_states)
        self.ssm["obs_cov"] = np.array([[0.]])
        self.ssm['obs_intercept'] = np.zeros ((1,self.nobs), dtype=dtype)

        self.ssm['state_intercept'] = np.zeros ((4,self.nobs), dtype=dtype)
        for ib in range(self.nb):
            self.ssm['state_intercept'][:,ib*self.nt-1] = \
                np.flip(self.initial_cases[ib,0:self.cut+1])
        self.ssm["selection"] = np.array([[1.],[0.],[0.],[0.]])
        if self.fix_vac: # Model VAC: corrected vaccine effect
            self.ssm["transition"][0,3,:] = \
                np.exp (np.dot(self.exog[:,:self.nn-self.nb-3],         params[:self.nn-self.nb-3]) + \
                        np.dot(self.exog[:, self.nn-self.nb-2:self.nn], params[ self.nn-self.nb-2:self.nn]) + \
                        self.delta_r_var) * \
                    ((1.0-self.exog[:,self.nn-self.nb-3])+(1.0-params[self.nn-self.nb-3])*self.exog[:,self.nn-self.nb-3])
        else: # Model DYN: Vaccine effect as in StopptCOVID
            self.ssm["transition"][0,3,:] = \
                np.exp (np.dot (self.exog,params[:self.nn]) + \
                        self.delta_r_var)
        self.ssm["transition"][1,0,:] = 1.0
        self.ssm["transition"][2,1,:] = 1.0
        self.ssm["transition"][3,2,:] = 1.0
        for i_b in range(self.nb):
            self.ssm["transition"][:,:,i_b*self.nt-1]=0.0
        self.ssm["state_cov"] = np.zeros ((1,1,self.nobs),dtype)
        self.ssm["state_cov"][0,0,:] = params[-1] * (1.0+self.endog[:,0])
                 
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
        params = super(renewal, self).update(params, **kwargs)

        if self.fix_vac: # Model VAC: corrected vaccine effect
            self.ssm["transition"][0,3,:] = \
                np.exp (np.dot(self.exog[:,:self.nn-self.nb-3],        params[:self.nn-self.nb-3]) + \
                        np.dot(self.exog[:, self.nn-self.nb-2:self.nn], params[ self.nn-self.nb-2:self.nn]) + \
                        self.delta_r_var) * \
                    ((1.0-self.exog[:,self.nn-self.nb-3])+(1.0-params[self.nn-self.nb-3])*self.exog[:,self.nn-self.nb-3])
        else: # Model DYN: Vaccine effect as in StopptCOVID
            self.ssm["transition"][0,3,:]=\
                np.exp (np.dot(self.exog,params[:self.nn]) + self.delta_r_var)
        for i_b in range(self.nb): # zero appropriate terms between time series
            self.ssm["transition"][:,:,i_b*self.nt-1] = 0.0
        self.ssm["state_cov"][0,0,:] = params[-1] * (1 + self.endog[:,0])

        

# Driver for fitting the model and computing the confidence intervals using
# a stationary bootstrap
class model:

    def __init__(self, x_in, case_in, y_in, w_in, delta_r_var, keys_in, params_in = None,
                 block_length = None, nboot = 100, maxiter_base = 50, maxiter_boot = 25, fix_vac = False):

        # x_in: nb*nt*nn array of explanatory variables where
        #       nb = number of geographical entities
        #       nt = number of time step
        #       nn = number of NPI variables + number of entities
        # case_in: case data to be fitted (nb*nt array)
        # y_in: R(t) as response variable for determining initial guess values
        #            for effects using WLS (nb*nt array)
        # w_in: weights: weights for WLS (nb*nt array)
        # delta_r_var: change of ln R(t) due to variants relative to wild type (nb*nt array)
        # keys_in: names of explanatory variables (nn array)
        # params_in: starting values for parameters, can be used for faster
        #            convergence
        # cut: number of data points to be cut from the time series, has
        # to be 3 for current version of this module, but is already included
        # to accommodate renewal models with a different infectivity function
        # block_length: average block length for stationary bootstrap, will be
        #               determined automatically if not set
        # nboot: number of bootstrap samples
        # maxiter_base: maximum number of iterations for obtaining the best-fit point estimate
        # maxiter_boot: maximum number of iterations during the bootstrap

        # Returned parameter estimates:
        # params[:nn-nb]: NPI regression coefficients
        # params[nn-nb:nn]: state fixed effects
        # params[nn]: scaling factor variance


        print ("Estimating NPI effects based on renewal equation.")
        print ("Confidence intervals based on stationary bootstrap.")

        self.fix_vac = fix_vac
        nt = np.shape(x_in)[1]
        nb = np.shape(x_in)[0]
        nn = np.shape(x_in)[2]
        keys_x = keys_in
        cut = 3
        
        model_dyn = renewal (x_in, case_in, y_in, w_in, delta_r_var, keys_in, params_in = params_in,\
                             cut = cut, fix_vac = fix_vac)
        res_dyn = model_dyn.fit(maxiter = maxiter_base)
        params = res_dyn.params
        cpred = np.float32(np.reshape(res_dyn.fittedvalues,(nb,nt-cut)))
        resid = np.reshape(res_dyn.resid,(nb,nt-cut))
        resid = resid / np.sqrt (case_in[:,cut:]+1)
        dparam = []
        params_boot = []
        resid = resid[:,1:] # First residual will be zero by construction
        cpred = cpred[:,1:] # First residual will be zero by construction

        
        if block_length == None:
          # Determine optimal block length for stationary bootstrap.
          # Use in-built function from ARCH package on residuals in each state
          # and then take the average value.
          restest = np.reshape(resid[:,:],(nb,nt-cut-1))
          #print(abs.optimal_block_length(restest[i,:]))
          opt_len = np.zeros(nb)
          for i in range(nb):
            opt_len[i] =  abs.optimal_block_length(restest[i,:])['stationary']
          block_length = np.round (np.average (opt_len))
          print ("Average block length not specified, using length of %i." % block_length)
        else:
          print ("Using specified block length of %i." % block_length)

        # ----------------------------------------------------------------------
        # Perform the bootstrap
        sbs = abs.StationaryBootstrap(block_length,np.arange(nt-cut-1))

        ctest = np.copy (case_in)

        print ("Boostrapping, resampling %i times." % nboot)
        for data,inx in tqdm(sbs.bootstrap(nboot)):
            ii=data[0].astype(int)
            # Boostrap residuals
            ctest[:,cut+1:] = np.exp((np.log(cpred[:,:]/case_in[:,cut+1:])*np.sqrt(case_in[:,cut+1:]))[:,ii] /
                                     np.sqrt(case_in[:,cut+1:]+1))*case_in[:,cut+1:]
            # Alternative resampling method:
            #ctest[:,cut+1:] = np.float32 (case_in[:,cut+1:] + np.sqrt(case_in[:,cut+1:]) *
            #                             np.minimum (np.maximum (resid[:,ii],
            #                                                     -0.8 * np.sqrt (case_in[:,cut+1:])), 0.8 * np.sqrt (case_in[:,cut+1:])))    
            res_tmp =  renewal (x_in, ctest, y_in, w_in, delta_r_var, keys_in,
                                params_in = params, cut = cut, fix_vac = fix_vac).fit(maxiter = maxiter_boot)
            dparam.append([res_tmp.params-params])
            params_boot.append([res_tmp.params])

        dparam = np.reshape (np.array(dparam), (nboot,nn+1))
        params_boot = np.reshape (np.array(params_boot), (nboot,nn+1))

        self.dparams = dparam # Store this in case we need the actual distribution
        self.params_boot = params_boot # Store parameter values as well
        self.params = params
        if self.fix_vac:
            # Transform \eta_vac to 1-\eta_vac
            self.params_boot [:,nn-nb-3] = np.log (1.0 - params_boot [:,nn-nb-3])
            self.params [nn-nb-3] = np.log (1.0 - params [nn-nb-3])

        # 95% confidence interval from standard deviation of bootstrap sample
        self.err = np.sqrt (np.average (dparam[:,:]**2, axis=0))*1.96
        self.result = res_dyn
        
    def conf_int(self):
        # Method for returnting confidence intervals
        return np.stack([self.params-self.err,self.params+self.err],axis=1)

