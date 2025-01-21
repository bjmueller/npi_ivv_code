# ==============================================================================
# Auxiliary plotting module
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics as smgraphics
from scipy.signal import savgol_filter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import cycle


def plot_results(models, nnpi=54, xlim=(0.4,1.22), xticks=np.arange(0.7,1.2,0.1), \
                 plot_log=False, labels=[], tag='', ybuf=0, keys=[]):

    # Generates a figure of point estimates and confidence intervals for one or multiple models
    
    # models:   list of objects of class regression_result
    # nnpi:     number of NPIs to consider
    # xlim:     range of effect sizes to plot
    # xticks:   tick marks for x-axis if these need to specified manually
    # plog_log: directly plot effect on ln R(t) if plot_log=False, otherwise
    #           plot multiplier for R(t)
    # labels:   list of labels for models to be used in legend
    # tag:      tag for PDF file names
    # ybuf:     buffer space on the y-axis to optimise the layout
    # keys:     list of names of explanatory variables
    
    nargs = len(models)
    plt.figure(figsize=(6.4,18))
    cnt = -nargs/2+0.5
    for model in models:
        offs = 0.1 * cnt
        cnt += 1
        if plot_log:
            y=model.params[0:nnpi]
            errpl=y-model.conf_int[0:nnpi,1]
            errmn=model.conf_int[0:nnpi,0]-y
        else:
            y=np.exp(model.params[0:nnpi])
            errpl=np.exp(model.conf_int()[0:nnpi,1])-y
            errmn=-np.exp(model.conf_int()[0:nnpi,0])+y
        plt.errorbar(y,-np.arange(nnpi)+offs,xerr=np.array([errmn,errpl]),fmt='o')
    if plot_log:
        plt.xlabel(r'$\Delta \ln \mathcal{R}$')
    else:
        plt.xlabel(r'Effect on $\mathcal{R}$')
    plt.ylim(-nnpi-0.2-ybuf,0.2)
    plt.yticks(-np.arange(nnpi),labels=keys,rotation=0,minor=False)
    plt.xticks(xticks)
    plt.minorticks_on()
    plt.xlim(xlim)
    if labels != None:
        plt.legend(labels,ncol=1,loc='upper left')
    plt.axvline(1,linestyle='dashed')
    if tag=='':
        plt.savefig('npi_effects.pdf', bbox_inches = None)
    else:
        plt.savefig('npi_effects_'+tag+'.pdf', bbox_inches = None)


def plot_results_split(models, nnpi=54, xlim=(0.4,1.22), xticks=np.arange(0.7,1.2,0.1),\
                       plot_log=False, labels=[], tag='', ybuf=0, keys=[],\
                       nsplit=2, colors=None):

    # Generates a figure of point estimates and confidence intervals for one or multiple models
        
    # models:   list of objects of class regression_result
    # nnpi:     number of NPIs to consider
    # xlim:     range of effect sizes to plot
    # xticks:   tick marks for x-axis if these need to specified manually
    # plog_log: directly plot effect on ln R(t) if plot_log=False, otherwise
    #           plot multiplier for R(t)
    # labels:   list of labels for models to be used in legend
    # tag:      tag for PDF file names
    # ybuf:     buffer space on the y-axis to optimise the layout
    # keys:     list of names of explanatory variables
    # nsplit:   split plot into nsplit separate panels
    # colors:   color sequence, default sequence will be used if colors=None

    nargs = len(models)

    for iplot in range(nsplit):
        n_start = iplot * (nnpi//nsplit)
        n_end = (iplot+1) * (nnpi//nsplit)
        #if iplot==nsplit-1: n_end = nnpi+1
        plt.figure(figsize=(6.4,10))
        cnt = -nargs/2+0.5
        if colors is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            color = cycle (prop_cycle.by_key()['color'])
        else:
            color = cycle (colors)
        for model in models:
            offs = 0.1 * cnt
            cnt += 1
            if plot_log:
                y=model.params[n_start:n_end]
                errpl=y-model.conf_int[n_start:n_end,1]
                errmn=model.conf_int[n_start:n_end,0]-y
            else:
                y=np.exp(model.params[n_start:n_end])
                errpl=np.exp(model.conf_int()[n_start:n_end,1])-y
                errmn=-np.exp(model.conf_int()[n_start:n_end,0])+y
            plt.errorbar(y,-np.arange(n_end-n_start)+offs,xerr=np.array([errmn,errpl]),fmt='o',color=next(color))
            #plt.minorticks_on()
            #plt.ylabel('NPI type')
        if plot_log:
            plt.xlabel(r'$\Delta \ln \mathcal{R}$')
        else:
            plt.xlabel(r'Effect on $\mathcal{R}$')
            plt.ylim(n_start-n_end-0.2-ybuf,0.2)
        plt.yticks(-np.arange(n_end-n_start),labels=keys[n_start:n_end],rotation=0,minor=False)
        plt.xticks(xticks)
        plt.minorticks_on()
        plt.xlim(xlim)
        if labels != None and iplot==0:
            plt.legend(labels[n_start:n_end],ncol=1,loc='upper left')
        plt.axvline(1,linestyle='dashed')
        
        if tag=='':
            plt.savefig('npi_effects_'+str(iplot)+'.pdf')
        else:
            plt.savefig('npi_effects_'+tag+'_'+str(iplot)+'.pdf')
        


def plot_vif (xin, n_npi, keys_npi):

    # Plot variance inflation factors for NPIs
    
    # x_in:     nb*nt*nn array of explanatory variables where
    #           nb = number of geographical entities
    #           nt = number of time step
    #           nn = number of NPI variables + number of entities
    # n_npi:    compute and plot variance inflation factors for first n_npi explanatory variables
    # keys_npi: names of explanatory variables
    
    vif = np.zeros(n_npi)
    nt = np.shape(xin)[0]
    nb = np.shape(xin)[1]
    for i in range (n_npi):
        vif[i] = variance_inflation_factor(np.reshape(xin[:,:,:n_npi],[nb*nt,n_npi]), i)
        #print(keys_npi[i],vif[i],vif[i]<10.)
    plt.figure(figsize=(6.4,18))
    plt.errorbar(vif,-np.arange(n_npi),xerr=[vif-1.,vif*0],fmt='o')
    plt.ylabel('NPI type')
    plt.yticks(-np.arange(n_npi),labels=keys_npi[:n_npi],rotation=0,minor=False)
    plt.ylim(-n_npi+0.5,0.5)
    plt.xscale('log')
    plt.xlabel('Variance inflation factor')
    plt.xlim(1,2e2)
    plt.axvline(10,linestyle='dotted')
    plt.savefig('vif.pdf')
    #print(keys[np.where(vif<10.)[0]])


def plot_fit (results, yin, ncase, date, labels, alpha = 0.0, delta = 0.0, tag = ''):

    # Plot prediction vs. data for national average of R(t) for a single model
    
    # results: Object of class RegressionResults, or similar structure.
    #          results.fittedvalues should contain the fit to log R(t).
    # yin:     Data for log R(t), shape (nb,nt).
    # ncase:   Case numbers per day for each state (up to a *common*
    #          scaling factor), shape (nb,nt).
    # date:    Arrays of dates, shape (nt).
    # labels:  Label of sub-entities (array of strings)
    # alpha:   Share of alpha variant, array of shape (nb,nt)
    # delta:   Share of delta variant, array of shape (nb,nt)
    # tag:     Tag for PDF file names

    plt.clf()
    nb = np.shape(yin)[0]
    nt = np.shape(yin)[1]
    ydata = yin + 0.3*alpha + 0.6*delta
    plt.figure(figsize=(6.4,4.8))
    rt_av = np.sum(np.exp(ydata)*ncase,0)/np.sum(ncase,0)
    plt.plot(date,rt_av,linewidth=1)
    ypred = np.reshape(results.fittedvalues,[nb,nt]) + 0.3*alpha + 0.6*delta
    ypred_av = np.sum(np.exp(ypred)*ncase,0)/np.sum(ncase,0)
    plt.plot(date,ypred_av,linewidth=1)
    plt.xlabel(r'Julian date')
    plt.ylabel(r'$\mathcal{R}(t)$')
    plt.minorticks_on()
    plt.ticklabel_format(style = 'plain')
    if tag=='':
        plt.savefig('fit_vs_data_national_rt.pdf')
    else:
        plt.savefig('fit_vs_data_national_rt_'+tag+'.pdf')
    plt.tight_layout()
    plt.show()

    # Plot prediction vs. data for R(t) at the state level
    f, ax = plt.subplots (4,4,figsize=(6.4*4,4.8*4), sharex=True, sharey=True)
    f.subplots_adjust (hspace=0.0, wspace=0.0)
    #ypred = np.reshape (results.fittedvalues,[nb,nt]) + 0.3*alpha + 0.6*delta
    for ib in range(nb):
        ax[ib//4,ib%4].plot(date,np.exp(ydata[ib,:]),linewidth=1)  
        ax[ib//4,ib%4].plot(date,np.exp(ypred[ib,:]),linewidth=1)
        ax[ib//4,ib%4].legend(labels=['data','fit'])
        ax[ib//4,ib%4].tick_params(axis='both',which='major',direction='inout',\
                                   right=True,top=True,length=6)
        ax[ib//4,ib%4].tick_params(axis='both',which='minor',direction='inout',\
                                   right=True,top=True,length=3.5)
        ax[ib//4,ib%4].minorticks_on()
        ax[ib//4,ib%4].ticklabel_format(style = 'plain')
        if ib//4==3: ax[ib//4,ib%4].set_xlabel(r'Julian date')
        if ib%4==0: ax[ib//4,ib%4].set_ylabel(r'$\mathcal{R}(t)$')
        plt.text(0.05, 0.91, labels[ib], transform=ax[ib//4,ib%4].transAxes)
    if tag=='':
        plt.savefig('fit_vs_data_states_rt.pdf')
    else:
        plt.savefig('fit_vs_data_states_rt_'+tag+'.pdf')
    plt.show()

    # Plot residuals for log R(t) at the state level
    f, ax = plt.subplots (4,4,figsize=(6.4*4,4.8*4), sharex=True, sharey=True)
    f.subplots_adjust (hspace=0.0, wspace=0.0)
    ypred = np.reshape(results.fittedvalues,[nb,nt])
    for ib in range(nb):
        ax[ib//4,ib%4].plot(date,(ypred-yin)[ib,:],linewidth=1)
        ax[ib//4,ib%4].tick_params(axis='both',which='major',direction='inout',\
                                   right=True,top=True,length=6)
        ax[ib//4,ib%4].tick_params(axis='both',which='minor',direction='inout',\
                                   right=True,top=True,length=3.5)
        ax[ib//4,ib%4].minorticks_on()
        ax[ib//4,ib%4].ticklabel_format(style = 'plain')
        if ib//4==3: ax[ib//4,ib%4].set_xlabel(r'Julian date')
        if ib%4==0: ax[ib//4,ib%4].set_ylabel(r'$\delta \log \mathcal{R}(t)$')
        plt.text(0.95, 0.91, labels[ib], transform=ax[ib//4,ib%4].transAxes, \
                 horizontalalignment='right')
    if tag=='':
        plt.savefig('residuals.pdf')
    else:
        plt.savefig('residuals_'+tag+'.pdf')


def plot_fit_multiple (models, yin, ncase, date, labels, alpha = 0.0, delta = 0.0, tag = ''):

    # Plot prediction vs. data for national average of R(t) for multiple models
    
    # results: Object of class RegressionResults, or similar structure.
    #          results.fittedvalues should contain the fit to log R(t).
    # yin:     Data for log R(t), shape (nb,nt).
    # ncase:   Case numbers per day for each state (up to a *common*
    #          scaling factor), shape (nb,nt).
    # date:    Arrays of dates, shape (nt).
    # labels:  Labels for models
    # alpha:   Share of alpha variant, array of shape (nb,nt)
    # delta:   Share of delta variant, array of shape (nb,nt)
    # tag:     Tag for PDF file names
    
    # Plot prediction vs. data for national average of R(t)
    plt.clf()
    nb = np.shape(yin)[0]
    nt = np.shape(yin)[1]
    ydata = yin + 0.3*alpha + 0.6*delta
    plt.figure(figsize=(6.4,4.8))
    rt_av = np.sum(np.exp(ydata)*ncase,0)/np.sum(ncase,0)
    plt.plot(date,rt_av,linewidth=1)
    for results in models:
        ypred = np.reshape(results.fittedvalues,[nb,nt]) + 0.3*alpha + 0.6*delta
        ypred_av = np.sum(np.exp(ypred)*ncase,0)/np.sum(ncase,0)
        plt.plot(date,ypred_av,linewidth=1)

    plt.legend(['data']+labels)
    plt.xlabel(r'Julian date')
    plt.ticklabel_format(style = 'plain')
    plt.ylabel(r'$\mathcal{R}(t)$')
    plt.minorticks_on()
    if tag=='':
        plt.savefig('fit_vs_data_national_rt.pdf')
    else:
        plt.savefig('fit_vs_data_national_rt_'+tag+'.pdf')
    plt.tight_layout()
    plt.show()
        
