from __future__ import division, unicode_literals, absolute_import, print_function
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import os, argparse, sys, re, warnings
from dateutil.relativedelta import relativedelta, weekday
from datetime import datetime, timedelta
import scipy.optimize as optimize
from scipy import stats
import seaborn as sns
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot
#from sklearn.experimental import enable_halving_search_cv
#from sklearn.model_selection import ParameterGrid, GridSearchCV, HalvingGridSearchCV
from sklearn.model_selection import ParameterGrid, GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandablob
from azure.storage.blob import ContainerClient
import dask.dataframe as dd
from dask.dataframe import from_pandas
import psycopg2, pyodbc
import mysql.connector
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv(dotenv_path='/home/parallels/.env')
container = ContainerClient.from_connection_string(conn_str=os.getenv('azure_connect_str'), container_name=os.getenv('azure_top_level_container_name'))
today = (datetime.today()).replace(hour=0, minute=0, second=0, microsecond=0)
yesterday = today - timedelta(days=1)

def string_float():
    """Action for argparse that allows a mandatory and optional
    argument, a string and floats, with default for the floats
    """
    class StringFloat(argparse.Action):
        """Action to assign a string and optional integer"""
        def __call__(self, parser, namespace, values, option_string=None):
            values[0] = values[0].upper()
            if values[0] not in ('AE','APE'):
                raise argparse.ArgumentError(self, ('first argument to "{}" must be AE or APE'.format(self.dest)))
            if len(values) > 3:
                raise argparse.ArgumentError(self,'argument "{}" requires not more than 3 arguments'.format(self.dest))
            for i in range(1,len(values)):
                if i == 1:
                    arg = 'second'
                else:
                    arg = 'third'
                try:
                    values[i] = float(values[i])/100
                except ValueError:
                    raise argparse.ArgumentError(self, (arg+' argument to "{}" requires '
                               'a valid numeric value'.format(self.dest)))
                if not (0 <= values[i] <= 1):
                    raise argparse.ArgumentError(self, (arg+' argument to "{}" '
                               'must be in the range 0-100'.format(self.dest)))
            # if not given use default values    
            if len(values) < 2:
                values.append(0.2)
                values.append(0.5)
            elif len(values) < 3:
                values.append(0.5)
            setattr(namespace, self.dest, values)
    return StringFloat

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive value" % value)
    return value

def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError("Not a valid date: '{0}'".format(s))

def two_args_float_bool():
    class FloatBool(argparse.Action):
        """Action to assign a string and optional integer"""
        def __call__(self, parser, namespace, values, option_string=None):
            values[1] = values[1].lower()
            try:
                values[0] = float(values[0])
            except ValueError:
                raise argparse.ArgumentError(self, ('first argument to "{}" requires '
                                                    'a float'.format(self.dest)))
            if not (0 <= values[0] <= 1):
                    raise argparse.ArgumentError(self, ('first argument to "{}" '
                               'must be in the range 0-1'.format(self.dest)))
            if values[1] not in ['true','false']:
                raise argparse.ArgumentError(self, ('second argument to "{}" requires '
                                                    'a bool'.format(self.dest)))
            else:
                if values[1] == 'true':
                    values[1] = True
                else:
                    values[1] = False
            setattr(namespace, self.dest, values)
    return FloatBool

parser = argparse.ArgumentParser(description='Occupancy Forecast & suggested rate change')
group = parser.add_mutually_exclusive_group()
parser.add_argument('-hid','--fucsia_id', type=check_positive, required=True, help="Hotel Fucsia identifier")
parser.add_argument('-s','--startdate', type=valid_date, default=today ,help="Start Date - format YYYY-MM-DD")
group.add_argument('-n','--nday', type=check_positive, default=180, help="Number of days to process")
group.add_argument('-e','--enddate', type=valid_date, help="End Date - format YYYY-MM-DD")
group.add_argument('-cvd','--covid_dates', default=[pd.to_datetime('2020-03-01'),pd.to_datetime('2021-12-31')], type=valid_date, nargs=2, metavar=('covid_start_date','covid_end_date'), help="Covid start and end dates - format YYYY-MM-DD")
parser.add_argument('-g','--graph', action="store_true", help="Produce some graphs and exit")
parser.add_argument('-gs','--gridsearch', action="store_true", help="Parameter estimation using grid search with cross-validation")
parser.add_argument('-cv','--crossval',nargs='+',action=string_float(),metavar=('error_type','params',),
                    help="Evaluate forecast error with cross-validation (max number of days in the future we can consider to achieve, on average, an accuracy below threshold for at least q%% forecasts). Mandatory param: error type [AE or APE], optional params: threshold and q-quantile")
parser.add_argument('-psd','--prev_sd', default=15, type=check_positive, help="Minimum number of days to start the forecast approach based on previous stay-dates final occupancies")
parser.add_argument('-cfg','--config', default=0, type=check_positive, help="Parameter configuration identifier")
group_rc = parser.add_argument_group('RC')
group_rc.add_argument('-rch','--rate_channel', default='bcom',choices=['min', 'bcom', 'hcom'], help="Rate channel")
group_rc.add_argument("-mrp", "--market_rate_par", nargs=2, default=[0.,True], action=two_args_float_bool(),
                      help="minimum fraction of days of availabiliy for market rate evaluation and application of Prophet to interpolate data",metavar=("min_frac_avail","interpolation"))
group_rc.add_argument('-elm','--elastmodel',default='glog',choices=['glog', 'log'], help='Select elasticity model type')
group_rc.add_argument('-elf','--elastfactor',default=1,metavar=('elasticity_multiplier'), type=check_positive_float, help='Scaling factor to be applied to price elasticity (to scale it up if >1 or down if <1)')
group_rc.add_argument('-elp','--elastpar', default=[1.,1.], nargs=2, metavar=('rate_disp_nsigma','occ_gain_lr'), type=check_positive_float, help='N. sigmas to define rate dispersion and expected occupancy gain factor at very low rate')
group_rc.add_argument('-rcp','--ratechangepar', default=[1.1,0.9], nargs=2, metavar=('cap_multiplier','floor_multiplier'), type=check_positive_float, help='Scaling factors to be applied to rate bounds (cap, floor) to scale them up if >1 or down if <1')
group_rc.add_argument('-rct','--ratechangethr', nargs=2, metavar=('min_thr', 'max_thr'), type=check_positive_float, help='Min and max rate change thresholds')
group_rc.add_argument('-rcl','--ratechangelink', default=1., metavar=('inc'), type=check_positive_float, help='Suggestion linked to market rate (ignoring demand forecast) increased by inc (inc=1 => no increment)')
args = parser.parse_args()

fucsia_id = args.fucsia_id
if args.enddate:
    args.nday = 1+(args.enddate-args.startdate).days
else:
    args.enddate = args.startdate + timedelta(days=args.nday-1)
if args.nday > 180:
    parser.error("NDAY should be <= 180")
if args.startdate and args.enddate and args.startdate > args.enddate:
    parser.error("ENDDATE should be >= than STARTDATE")
if args.covid_dates[0] >= args.covid_dates[1]:
    parser.error("Covid end-date should be > than Covid start-date")    
if args.ratechangelink and args.ratechangethr:
    parser.error("-rcl and -rct are mutually exclusive")

ProdGraph = args.graph
GridSearch = args.gridsearch
CrossValPar = args.crossval
RateChangePar = args.ratechangepar
RateChangeThr = args.ratechangethr
RateChangeLink = args.ratechangelink
RateChannel = args.rate_channel
MarketRatePar = args.market_rate_par
ElastModel = args.elastmodel
ElastFactor = args.elastfactor
ElastPar = args.elastpar
first_fcst_day = args.startdate
end_day = args.enddate
next_days_to_fcst = args.nday
PsdNdayThr = args.prev_sd
CovidDates = args.covid_dates
cfg_ver = args.config

if first_fcst_day >= today:
    last_book_day = (today - timedelta(days=1))
    day_of_fcst = today
else:
    last_book_day = (first_fcst_day - timedelta(days=1))
    day_of_fcst = first_fcst_day

if next_days_to_fcst is None:
    next_days_to_fcst = (end_day-first_fcst_day).days+1
    max_dates_to_fcst = pd.date_range(start=first_fcst_day, end=end_day)
else:
    max_dates_to_fcst = pd.date_range(start=first_fcst_day, periods=next_days_to_fcst)

dir='/home/parallels/blastness/rm/pro/azure/'
dirout_rms = dir+'fcst/'
#dirout_rms = '/home/blastness/Dropbox/Fox Sibylla/Output/C'+str(cfg_ver)+'/RMS/'
#dirout_rms = '/home/blastness/Dropbox/Fox Sibylla/Output/Pro/'
sns.set(style="whitegrid")
MO, TU, WE, TH, FR, SA, SU = weekdays = tuple(weekday(x) for x in range(7))
# Random Forest: parametri e caratteristiche
params_bar = {'n_estimators':100,'criterion':'mae','max_depth':13,'max_features':'sqrt','min_samples_leaf':3,'random_state':0}
PCAtrans = False
if cfg_ver==0:
    features = ['lat','long','stars','br_services','br_clean','br_comfort','br_value','br_location']
elif cfg_ver==5:
    features = ['lat','long','stars','br_services','br_clean','br_comfort','br_value','br_location']
elif cfg_ver==6:
    features = ['lat','long','stars','br_services','br_clean','br_comfort','br_value','br_location']
    PCAtrans = True
elif cfg_ver==7:
    features = ['lat','long','stars','br_services','br_clean','br_comfort','br_value','br_location','type_city','type_countryside','type_lake','type_mountain','type_other','type_sea']
    PCAtrans = True
elif cfg_ver>=8:
    features = ['lat','long','stars','br_services','br_clean','br_comfort','br_value','br_location']
    PCAtrans = True
