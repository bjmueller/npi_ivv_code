# ==============================================================================
# Module for ranking the robustness of effects in the model ensemble
# ==============================================================================

from scipy.stats import t
import numpy as np

def rank_effects (models, keys, nparams, dof, cut1 = 2, cut2 = 0.1, latex_out = False):

    # models: list of models
    # keys: list of names of explanatory variables
    # nparams: consider first naparams explanatory variables
    # dof: degrees of freedom to be used in t-distribution
    # cut1: cut-off for showing integer scores
    # cut2: cut-off for showing false-positive risk score
    # latex_out: output in LaTeX table format if latex_out=True
    
    n_models = len(models)

    score1 = np.zeros(nparams,np.int32) # Score 1
    score1a = np.zeros(nparams,np.int32) # Number of models that disagree with median estimate in sign
    score1b = np.zeros(nparams,np.int32) # Number of models whose CI for a parameter overlaps with zero
    score2 = np.zeros(nparams) # False-positive risk score
    med = np.zeros(nparams) # Median effect estimates across ensemble
    
    for i_npi in range (nparams):
        params_all = []
        conf_hi = []
        conf_lo = []
        for model in models:
            params_all.append (model.params[i_npi])
            conf_hi.append (model.conf_int()[i_npi,1])
            conf_lo.append (model.conf_int()[i_npi,0])
        params_all = np.array(params_all)
        conf_hi = np.array(conf_hi)
        conf_lo = np.array(conf_lo)
        med[i_npi] = np.median(params_all)
        # Count models that disagree about the sign
        score1a[i_npi] = int(np.sum(np.heaviside(-params_all*med[i_npi],0)))
        # Count CIs overlapping with zero
        score1b[i_npi] = int(np.sum(np.heaviside(-conf_hi*conf_lo,0)))
        score1[i_npi] = score1a[i_npi] + score1b[i_npi]
        
    i_sorted = np.argsort(score1+1e-2*score1a+1e-4*score1b+1e-6*med)

    for i_npi in i_sorted:
        # Only output interventions that reduce R(t), except for seasonality
        if score1[i_npi] <= cut1 and (med[i_npi]<0 or i_npi>=nparams-2):
            if latex_out:
                print("%s & %i & %i & %i \\\\" % (keys[i_npi],score1a[i_npi], score1b[i_npi],score1[i_npi]))   
            else:
                print(keys[i_npi], score1a[i_npi],score1b[i_npi], score1[i_npi])
    print("\n")
        
    for i_npi in range(nparams):
        for model in models:
            score2[i_npi] += t.cdf(0,dof,loc=np.sign(med[i_npi])*model.params[i_npi],\
                                   scale=(model.conf_int()[i_npi,1]-model.params[i_npi])/1.96)
        score2[i_npi] = score2[i_npi] / n_models

    i_sorted = np.argsort(score2)
    for i_npi in i_sorted:
        # Only output interventions that reduce R(t), except for seasonality
        if score2[i_npi] <= cut2 and (med[i_npi]<0 or i_npi>=nparams-2):
            if latex_out:
                print("%s & %3.3f \\\\" % (keys[i_npi],score2[i_npi]))
            else:
                print("%s %3.3f" % (keys[i_npi],score2[i_npi]))
