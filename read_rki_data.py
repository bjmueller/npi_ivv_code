# ==============================================================================
# Module for reading input data
# ==============================================================================

import numpy as np
import pandas as pd
import pyreadr

def hello():
  print("hello")
  return 22

def rki_data (path, file = 'data_main.rData'):

  # This function reads the response variable and the explanatory
  # variables underlying the baseline model of StopptCOVID,
  # plus some data not used in the baseline model.
  # The data are assumed to have been exported after the appropriate
  # best-fit lags have been applied to the explanatory variables.

  # INPUT
  # path: path to rData file containing the data from StopptCOVID
  # file: name of rData file

  # OUTPUT - tuple containing
  # 
  # 

  rki_main = pyreadr.read_r (path+'/'+file)
  rki_main = rki_main ['data_main']


  # Times of data points

  # Read date column and convert to Julian date
  #date_rd = pd.to_datetime(rki_main.date.to_numpy()).to_julian_date()
  date_rd = pd.to_datetime(rki_main['date'].to_numpy()).to_julian_date()
  nt = int(max(date_rd)-min(date_rd)) # number of dates
  print("Data from:", min(rki_main.date),min(date_rd))
  print("Data until:", max(rki_main.date),max(date_rd))
  print("Time series contaits %i days." % nt)

  # Convert date to real variable (Julian date) for easier handling.
  date_start = min(date_rd)
  date_end   = max(date_rd)
  # Create contiguous date array
  date = np.arange(date_start,date_end+1)
  date_01012020 = pd.to_datetime('01-01-2020').to_julian_date()

  # Number of times
  nt = (max(rki_main['date']) - min(rki_main['date'])).days + 1 # Number of time steps

  # Number of geographic entities (states - "Bundeslaender")
  nb = len(rki_main['Id_BL'].unique()) # Number of states, based on unique state IDs Id_BL
  # Find names of states
  keys_state = [] 
  for ib in range(nb): 
    keys_state.append(rki_main[rki_main['Id_BL']==ib+1]['BL'].iloc[0])


  # Now copy required data into temporary arrays (suffix _rd)
  # before we reshape the data based on the dimensions of time and
  # geographic entity.
  
  # Response variable: R(t) from StopptCOVID data set
  r_rd = rki_main['R']

  # Weights for WLS: Smoothed case data
  wt_rd = rki_main['N_smooth']

  # Unique ID for states (starts at 1!)
  Id_BL = rki_main['Id_BL']

  # Explanatory variables - relevant keys from StopptCOVID data set.
  # cos and sin component for seasonality will be introduced later based on the day of year.
  keys_x = \
    ['schools_measures_02',
     'schools_measures_03',
     'schools_measures_04',
     'schools_measures_05',
     'schools_measures_06',
     'private_space_measures_02',
     'private_space_measures_03',
     'private_space_measures_04',
     'private_space_measures_05',
     'workplace_measures_02',
     'daycare_measures_02',
     'daycare_measures_03',
     'daycare_measures_04',
     'public_space_measures_02',
     'public_space_measures_03',
     'public_space_measures_04',
     'public_space_measures_05',
     'public_event_outdoor_measures_02',
     'public_event_outdoor_measures_03',
     'public_event_outdoor_measures_04',
     'public_event_outdoor_measures_05',
     'public_event_outdoor_measures_06',
     'curfew_measures_02',
     'retail_measures_02',
     'retail_measures_03',
     'retail_measures_04',
     'retail_measures_05',
     'nightlife_measures_02',
     'nightlife_measures_03',
     'services_measures_02',
     'services_measures_03',
     'services_measures_04',
     'services_measures_05',
     'ceshg_max_measures_02',
     'ceshg_max_measures_03',
     'ceshg_max_measures_04',
     'ceshg_max_measures_05',
     'ceshg_max2_measures_02',
     'ceshg_max2_measures_03',
     'test_measures_02',
     'test_measures_03',
     'test_measures_04',
     'abstand_measures_02',
     'mask_measures_02',
     'mask_measures_03',
     'mask_measures_04',
     'mask_measures_05',
     'school_holiday',
     'after_holiday',
     'school_holiday_second_half',
     'easter_christmas',\
     'per_vacc_1']
     #'Impfquote_1'] # Alternative explanatory variable for vaccine effect

# Nicer labels for plots:
  keys_nice = [r'Schools L2',
   r'Schools L3',
  r'Schools L4',
  r'Schools L5',
  r'Schoole L6',
  r'Private spaces L2',
  r'Private spaces L3',
  r'Private spaces L4',
  r'Private spaces L5',
  r'Workplaces L2',
  r'Child care facilities L2',
  r'Child care facilities L3',
  r'Child care facilities L4',
  r'Public spaces L2',
  r'Public spaces L3',
  r'Public spaces L4',
  r'Public spaces L5',
  r'Public outdoor events L2',
  r'Public outdoor events L3',
  r'Public outdoor events L4',
  r'Public outdoor events L5',
  r'Public outdoor events L6',
  r'Stay-at-home orders L2',
  r'Retail L2',
  r'Retail L3',
  r'Retail L4',
  r'Retail L5',
  r'Night life L2',
  r'Night life L3',
  r'Service sector L2',
  r'Service sector L3',
  r'Service sector L4',
  r'Service sector L5',
  #sports, culture, tourism, restaurants
  r'CHRS, 1 at highest Lvl',
  r'CHRS, 2 at highest Lvl',
  r'CHRS, 3 at highest Lvl',
  r'CHRS, 4 at highest Lvl',
  r'CHRS, 1 at 2nd-highest Lvl',
  r'CHRS, 2 at 2nd-highest Lvl',
  r'COVID tests L2',
  r'COVID tests L3',
  r'COVID tests L4',
  r'Physical distancing L2',
  r'Masks L2',
  r'Masks L3',
  r'Masks L4',
  r'Masks L5',
  r'School holidays',
  r'After school holidays',
  r'School holidays (2nd half)',
  r'Easter \& Christmas',
  r'Vaccination (1st dose)',
  r'Seasonality ($\cos$)',
  r'Seasonality ($\sin$)',
  'bl1',
  'bl2',
  'bl3',
  'bl4',
  'bl5',
  'bl6',
  'bl7',
  'bl8',
  'bl9',
  'bl10',
  'bl11',
  'bl12',
  'bl13',
  'bl14',
  'bl15',
  'bl16']

  # Read data into numpy array
  x_rd = rki_main[keys_x].to_numpy()
  # Transform vaccination variable
  # x_rd[:,-1] = x_rd[:,-1]/100.
  # x_rd[:,-1] = -np.log(1.0-x_rd[:,-1])

  # Share of Alpha and Delta variant
  alpha_rd = rki_main['Alpha']
  delta_rd = rki_main['Delta']

  # Number of explanatory variables + fixed effects for states.
  nn = len(keys_x) + 2 + nb

  keys_x.extend(['cos','sin'])
  for i in range (nb):
      keys_x.append("bl%i" % (i+1))


  # Create arrays for response variable, explanatory variables and weights.
  y  = np.zeros((nb,nt))
  y [:,:] = np.NaN # Initialise with NaN to detect gaps in data
  x  = np.zeros((nb,nt,nn))
  alpha  = np.zeros((nb,nt))
  delta  = np.zeros((nb,nt))
  wt = np.zeros((nb,nt))

  # Populate arrays by looping through StopptCOVID data set (brute force solution
  #for i in range(len(date)):
  for i in range(len(date_rd)):    
     jj = Id_BL[i]-1
     ii = round(date_rd[i]-date_start)
     y[jj,ii] = np.log(r_rd[i])-0.3*alpha_rd[i]-0.6*delta_rd[i]
     alpha[jj,ii] = alpha_rd[i]
     delta[jj,ii] = delta_rd[i]
     x[jj,ii,:-2-nb] = x_rd[i]
     wt[jj,ii] = wt_rd[i]

  i_nan= np.where(np.isnan(y[:,:]))
    
  # Last two contain cos and sin basis functions for seasonality
  # Astronomical years has 365.2422 days
  x[:,:,-1-nb] = np.sin(2.0*np.pi*(np.tile(date,[nb,1])-date_01012020)/365.2422)
  x[:,:,-2-nb] = np.cos(2.0*np.pi*(np.tile(date,[nb,1])-date_01012020)/365.2422)

  # Imputation of missing data points:
  # Linear interpolation in gaps (interior) or constant extraploation (boundary)
  i_nan = np.where(np.isnan(y))
  if len(i_nan[0]>0):
    i_s = 1+np.where((i_nan[0][1:]!=i_nan[0][0:-1])|(i_nan[1][1:]!=i_nan[1][0:-1]+1))[0]
    i_s = np.insert(i_s,0,0)
    i_e = np.append(i_s[1:]-1,-1)
    #print(i_s)
    #print(i_e)
    for i_gap in range(len(i_s)):
      jj  = i_nan[0][i_s[i_gap]] 
      ii0 = i_nan[1][i_s[i_gap]]-1 
      ii1 = i_nan[1][i_e[i_gap]]+1
      print("Interpolating in state %i between data point %i and %i." % (jj, ii0, ii1))
      for ii in range(ii0+1,ii1):
        if ii0<0: ii0 = ii1
        if ii1>=nt: ii1 = ii0
        xi = (ii-ii0)/(ii1-ii0+2)
        y [jj,ii]   = (1.0-xi)*y [jj,ii0]  + xi*y [jj,ii1]
        x [jj,ii,:] = (1.0-xi)*x [jj,ii0,:]+ xi*x [jj,ii1,:]
        wt[jj,ii]   = (1.0*xi)*wt[jj,ii0]  + xi*wt[jj,ii1]
        print(ii,ii0,ii1,xi,y[jj,ii],y[jj,ii0],y[jj,ii1])
            
  # Matrix elemnts for fixed effects:
  for jj in range(nb):
      x[jj,:,-nb+jj] = 1.0
  #return 2
  #return y,x
  return y, x, wt, alpha, delta, date, keys_x, keys_nice, keys_state, nt, nn, nb


