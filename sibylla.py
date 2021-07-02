#!/home/parallels/anaconda3/envs/env/bin/python3
from sibylla_config import *
import sibylla_lib

sibylla_lib.Azure_Blobs_Download()
fox2_id = sibylla_lib.Get_Fox2_Id(fucsia_id)
# Ottengo i dati delle tariffe, il Competitiveness Index, le tariffe di mercato ed il Market Demand Index
rates, df_idx = sibylla_lib.Get_CompDemIdx(fucsia_id)
if len(rates) == 0:
    sibylla_lib.eprint("!!! Fucsia_id: ",fucsia_id,", nessun dato delle rates presente per le date selezionate !!!")
    RCData = False
elif len(df_idx) == 0:
    sibylla_lib.eprint("!!! Fucsia_id: ",fucsia_id,", dati Indice di competitivita e domanda non disponibili !!!")
    RCData = False        
else:
    RCData = True
# Ottengo i dati PMS
if fox2_id is not None:
    df_date, df_oday = sibylla_lib.Get_PMS(fox2_id)
    if len(df_date) < 30 or len(df_oday) < 30:
        sibylla_lib.eprint("!!! Fucsia_id: ",fucsia_id,", dati PMS insufficienti per le date selezionate !!!")
        PMSData = False
    elif df_date['date'].max() < last_book_day:
        sibylla_lib.eprint("!!! Fucsia_id: ",fucsia_id,", dati PMS non aggiornati al",last_book_day.strftime('%Y-%m-%d'))
        PMSData = False
    else:
        PMSData = True
else:
    PMSData = False

occ_opt_prices = pd.DataFrame({'rday':last_book_day.strftime('%Y-%m-%d'),'fucsia_id':fucsia_id,'Occupancy Forecast':np.nan,'Shopped Rate':rates['bar'].values.astype(float),'Market Gravity':df_idx['market_rate'].values.astype(float),'Gravity Attraction':df_idx['comp_idx'].values.astype(float),'Gravity Suggested Rate Change %':np.nan,'Occupancy Suggested Rate Change %':np.nan},index=max_dates_to_fcst)
occ_opt_prices.index.name = 'oday'

if PMSData:
    # Forecast basato sulle occupazioni delle date di soggiorno precedenti quelle di interesse    
    ndays_hist = (df_oday['oday'].max()-df_oday['oday'].min()).days
    ph = sibylla_lib.ProphetFcstPSD(df_oday,last_book_day,ProdGraph)
    if GridSearch:
        param_grid = {'yearly_seasonality': [True,20],
                      'weekly_seasonality': [True,20],
                      'seasonality_prior_scale':[10,20],
                      'growth': ['linear','logistic'],
                      'changepoint_prior_scale':[0.05,0.5,1]}
        if ndays_hist < 365:
            param_grid.update({'yearly_seasonality': [False],'growth': ['flat']})
        params = ph.GridSearchCV(param_grid,verbose=True)
    else:
        params = {'weekly_seasonality': True,
                  'yearly_seasonality': True,
                  'changepoint_prior_scale':0.5,
                  'seasonality_prior_scale': 20,
                  'growth': 'logistic'}
        if ndays_hist < 365:
            params.update({'yearly_seasonality': False,'growth': 'flat'})
    if CrossValPar:
        ph.CrossVal(params)
    occ_forecast = ph.Forecast(params)

    # Forecast basato sull'andamento temporale delle occupazioni delle date di soggiorno di interesse (considerato solo per i primi 'PsdNdayThr' giorni)   
    dfo = sibylla_lib.same_weekday_prev_years(df_date)
    last_occs = df_date[(df_date['oday'].isin(max_dates_to_fcst)) & (df_date['date']==max(df_date['date']))].drop('date',axis=1).set_index('oday')
    df_date['occupancy'] = df_date.groupby('oday')['occupancy'].apply(lambda x: x.diff().fillna(x))
    # le date ('oday') degli anni precedenti le sovrascrivo con quelle del forecast (next 180 days) in modo da associare ad esse la time-serie:
    # in particolare considero lo stesso giorno della settimana del forecast negli anni precedenti
    # eccetto che per date speciali per le quali utilizzo la stessa data (mm-dd) del forecast
    df_date = sibylla_lib.same_weekday_prev_years(df_date)
    not_null_occ = df_date[df_date['occupancy']!=0]['occupancy']
    median = not_null_occ.quantile(0.5)
    mad = stats.median_abs_deviation(not_null_occ,scale='normal')
    df_date['occupancy'] = np.where((df_date['occupancy']!=0) & (abs(df_date['occupancy']-median)/mad > 5),0,df_date['occupancy'])
    df_date = df_date.groupby('oday').apply(lambda x: x.reindex(pd.date_range(x.index.min(),last_book_day))).drop('oday',axis=1).rename_axis(['oday','date']).reset_index(level='oday')
    df_date['occupancy'].fillna(0,inplace=True)
    df_date['occupancy'] = df_date['occupancy'].cumsum()
    nyearly_cycles = len(df_date.index)/365
    # controllo di avere un numero sufficiente di dati PMS
    if nyearly_cycles < 2:
        sibylla_lib.eprint("!!! Fucsia_id: ",fucsia_id,", numero cicli annuali: ",'{0:.3g}'.format(nyearly_cycles),", inferiore al minimo !!!")
        exit(1)
    occ_forecast_otbd = pd.Series(name=occ_forecast.name,dtype=float)
    for fcst_day in list(max_dates_to_fcst[:PsdNdayThr].strftime('%Y-%m-%d')):
        ndays_to_fcst = int((pd.to_datetime(fcst_day)-(last_book_day+timedelta(days=1)))/np.timedelta64(1,'D'))+1    
        dates_to_fcst = pd.date_range(start=last_book_day+timedelta(days=1), periods=ndays_to_fcst)
        df_actual = dfo[dfo['oday']==fcst_day].drop('oday',axis=1).squeeze().rename(pd.to_datetime(fcst_day))
        dft = df_date[df_date['oday']==fcst_day].drop('oday',axis=1).squeeze().rename(pd.to_datetime(fcst_day))
        #first_book_day = pd.to_datetime(fcst_day)-relativedelta(years=3)+relativedelta(days=1)
        #dft = dft[dft.index >= first_book_day] # considera al max 3 anni di dati
        dft.dropna(inplace=True)
        dft = dft.where(dft>0,0)
        ndays_hist = (dft.index.max()-dft.index.min()).days
        ph = sibylla_lib.ProphetFcst(dft,last_book_day,last_occs,ProdGraph)
        if GridSearch:
            param_grid = {'yearly_seasonality': [True,20],
                      'weekly_seasonality': [False,True],
                      'seasonality_prior_scale':[10,20],
                      'growth': ['linear','logistic'],
                      'changepoint_prior_scale':[0.05,0.5,1]}
            if ndays_hist < 365:
                param_grid.update({'yearly_seasonality': [False]})
            params = ph.GridSearchCV(param_grid,verbose=True)
        else:
            params = {'weekly_seasonality': False,
                      'yearly_seasonality': True,
                      'changepoint_prior_scale':1,
                      'seasonality_prior_scale': 10,
                      'growth': 'logistic'}
            if ndays_hist < 365:
                params.update({'yearly_seasonality': False})
        if CrossValPar and fcst_day == first_fcst_day.strftime('%Y-%m-%d'):
            ph.CrossVal(params)
        if (dft == 0).all(): # avoid fitting if occupancy values are all zero
            unconst_fcst = pd.Series(data=0.,index=dates_to_fcst,name=fcst_day)
        else:
            unconst_fcst = ph.Forecast(params)
        if ProdGraph:
            fig=plt.figure(figsize=(16,8))
            fig.suptitle('Fucsia_id: '+str(fucsia_id)+' forecast del '+str(fcst_day),fontsize=16,fontweight='bold')
            sns.lineplot(data=pd.concat((df_actual.rename('Historical'),
                                         unconst_fcst.rename('Forecast')),axis=1,join='outer'),linewidth=2.5,dashes=False)
            plt.savefig(dir+'img/'+str('fcst_'+fcst_day))
            plt.show()
        final_ufcst = unconst_fcst.loc[fcst_day]
        occ_forecast_otbd = occ_forecast_otbd.append(pd.Series(name=occ_forecast_otbd.name,data=final_ufcst,index=[pd.to_datetime(fcst_day)]))
    occ_forecast.update(occ_forecast_otbd)
    occ_opt_prices['Occupancy Forecast'] = occ_forecast
    
# ---- suggest rates variation to maximize RevPar -----

if RCData:
    if PMSData:
        for day in rates.index:
            occupancy_otb = last_occs.loc[day].item()
            # starting prices
            rate0 = rates['bar'].loc[day]
            # Forecasted occupancy
            occupancy_fcst = occ_forecast.loc[day]
            cap = rates['cap'].loc[day]
            floor = rates['floor'].loc[day]
            # effective percentage capacity
            capacity = 100
            #dem_idx = df_idx['demand_idx'].loc[day]
            dem_idx = 1
            comp_idx = df_idx['comp_idx'].loc[day]
            if ElastModel == 'glog':
                gravity = df_idx['market_rate'].loc[day]
                rate_std = df_idx['comp_idx_std'].loc[day]*gravity
                rate_disp_width = ElastPar[0]*rate_std
                occupancy_max_thr = occupancy_fcst+(occupancy_fcst-occupancy_otb)*ElastPar[1]*comp_idx*dem_idx
                opt_results = optimize.minimize(sibylla_lib.generalised_logistic_demand, rate0, args=(rate0,occupancy_otb,occupancy_fcst,gravity,rate_disp_width,capacity,occupancy_max_thr),bounds=((floor, cap),),method='SLSQP')
            elif ElastModel == 'log':
                el_comp_slope = 1 # coeff. rapidita di diminuzione dell'elasticita allontanadosi dal prezzo di mercato
                el_comp_idx = np.exp(-(comp_idx-1)**2/(2*(df_idx['comp_idx_std'].loc[day]/el_comp_slope)**2))
                el_dem_idx = dem_idx
                elasticity = 0.05*ElastFactor*el_comp_idx*el_dem_idx
                opt_results = optimize.minimize(sibylla_lib.logistic_demand, rate0, args=(occupancy_fcst, elasticity, rate0, capacity, occupancy_otb), bounds=((floor, cap),), method='SLSQP')
            sugg_rate = opt_results['x'] # price maximizing RevPar
            sugg_rate_change = 100*(sugg_rate/rate0-1)
            if RateChangeThr is not None:
                min_rc_thr = RateChangeThr[0]
                max_rc_thr = RateChangeThr[1]
                sugg_rate_change = 0 if abs(sugg_rate_change) < min_rc_thr else sugg_rate_change
                sugg_rate_change = np.sign(sugg_rate_change)*max_rc_thr if abs(sugg_rate_change) > max_rc_thr else sugg_rate_change
            occ_opt_prices.loc[day,'Occupancy Suggested Rate Change %'] = sugg_rate_change
    link_rate = df_idx['market_rate']*RateChangeLink
    link_rate.loc[link_rate > rates['cap']] = rates['cap']
    link_rate.loc[link_rate < rates['floor']] = rates['floor']
    link_rate.index.name = 'oday'
    grav_sugg_rate_change = 100*(link_rate/rates['bar']-1)
    occ_opt_prices['Gravity Suggested Rate Change %'] = grav_sugg_rate_change.values.astype(float)

occ_opt_prices = (occ_opt_prices.reset_index(level=0))[['rday','fucsia_id','oday','Occupancy Forecast','Shopped Rate','Market Gravity','Gravity Attraction','Gravity Suggested Rate Change %','Occupancy Suggested Rate Change %']]
occ_opt_prices.to_csv(dirout_rms+"SibyllaRMS_V1.0_C"+str(cfg_ver)+"_"+str(fucsia_id)+"_"+last_book_day.strftime('%Y-%m-%d')+".csv",header=True,index=False,sep=",",float_format='%.2f')
