from sibylla_config import *

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))
    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

#--------------------------------------------------------------------------------------------------------------------

def Azure_Blobs_Download():
    for csv_file in ["DatiHotel_ML.csv","Reputations_ML.csv","Rates_ML.csv","PMS_ML.csv"]:
        blob_client = container.get_blob_client(blob="MLdata/"+csv_file)
        with open("./"+csv_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

#--------------------------------------------------------------------------------------------------------------------

def Get_Fox2_Id(fucsia_id):
    ids = pd.read_csv("./DatiHotel_ML.csv",sep=';',usecols=['idHotel','idHotelFox2'])
    ids = ids[ids['idHotel']==fucsia_id]
    if len(ids) == 0:
        eprint("!!! Fucsia_id: ",fucsia_id," non esistente !!!")
        exit(1)
    fox2_id = ids['idHotelFox2'].item()
    return fox2_id

#--------------------------------------------------------------------------------------------------------------------

def Covid_Range(ds):
    date = pd.to_datetime(ds)
    start_covid = CovidDates[0]
    end_covid = CovidDates[1]
    if date >= start_covid and date <= end_covid:
        return 1
    else:
        return 0
    
#--------------------------------------------------------------------------------------------------------------------
    
class ProphetFcstPSD:
    '''
    previsione di occupancy per una Stay-Date futura basata sull'andamento avuto nel passato per le Stay-Dates precedenti
    '''
    def __init__(self,row,last_book_day,ProdGraph):
        self.ndays_to_fcst = (end_day-last_book_day).days
        self.dates_to_fcst = max_dates_to_fcst
        self.df = row
        self.df.columns = ['ds','y']
        self.ProdGraph = ProdGraph
    def GridSearchCV(self,param_grid,verbose):
        dfp = self.df
        grid = ParameterGrid(param_grid)
        train_days = len(dfp)-self.ndays_to_fcst-10
        bparams={}
        for params in grid:
            model=Prophet(**params)
            dfp['cap'] = 100
            with suppress_stdout_stderr():
                model.fit(dfp)
                df_cv = cross_validation(model, initial=str(train_days)+' days', period='1 days', horizon = str(self.ndays_to_fcst)+' days')
            df_p = performance_metrics(df_cv,rolling_window=1,metrics=['mae'])
            bparams[(df_p['mae'].values).item()] = params
        best_params = bparams[min(bparams)]
        if verbose==True:
            for rmse, par in sorted(bparams.items()):
                print(par,": %6.3f" %(rmse))
            print("\n","Best parameters:")
            print(best_params,'\n')
        return best_params
    def CrossVal(self,params):
        err_type = CrossValPar[0]
        accuracy_thr = CrossValPar[1]
        quantile_val = CrossValPar[2]
        nfold = 100
        dfp = self.df
        horizon_days = self.ndays_to_fcst
        period_days = 1
        initial_days = len(dfp)-(horizon_days+period_days*nfold) 
        model=Prophet(**params)
        dfp['cap'] = 100
        with suppress_stdout_stderr():
            model.fit(dfp)
            df_cv = cross_validation(model, initial=str(initial_days)+' days', period=str(period_days)+' days', horizon=str(horizon_days)+' days')
        if err_type == 'APE':
            err_df = pd.DataFrame({'horizon':df_cv['ds']-df_cv['cutoff'],'ape':abs((df_cv['yhat']-df_cv['y'])/df_cv['y'])})
            err_quant = err_df.groupby('horizon').quantile(quantile_val)
            err_quant.rename(columns={'ape':'mdape'},inplace=True)
        elif err_type == 'AE':
            err_df = pd.DataFrame({'horizon':df_cv['ds']-df_cv['cutoff'],'ae':abs((df_cv['yhat']-df_cv['y'])/100)})
            err_quant = err_df.groupby('horizon').quantile(quantile_val)
            err_quant.rename(columns={'ae':'mdae'},inplace=True)
        max_days_quant = int(np.nan_to_num(err_quant[err_quant.le(accuracy_thr)].dropna().index.days.max()))
        if max_days_quant < self.ndays_to_fcst:
            if max_days_quant == 0:
                eprint("!!! Hotel_id: ",fucsia_id,": numero di giorni nel futuro entro cui l'errore del forecast viene stimato sotto soglia pari a 0 !!!")
                exit(1)
            else:
                eprint('Quantile: ',quantile_val,'max ndays: ',max_days_quant,'lower than',self.ndays_to_fcst,'requested!')
        if err_type == 'APE':
            fig = plot_cross_validation_metric(df_cv, metric='mdape',rolling_window=0)
        elif err_type == 'AE':
            fig = plot_cross_validation_metric(df_cv, metric='mae',rolling_window=0)
        plt.savefig(dir+'img/'+str('cross_validation_performance_'+str(fucsia_id)))
        err_quant['quantile'] = quantile_val
        err_quant.to_csv(dirout_rms+"prediction_error_psd_"+str(fucsia_id)+"_"+last_book_day.strftime('%Y-%m-%d')+".csv",header=True,index=True,sep=",",float_format='%.2f')
    def Forecast(self,params):
        holidays=pd.DataFrame(columns=['ds','holiday'])
        dft = self.df
        dft['covid'] = dft['ds'].apply(Covid_Range)
        model=Prophet(interval_width=0.95,changepoint_range=0.9,holidays=holidays,**params)
        model.add_regressor('covid')
        model.add_country_holidays(country_name='IT')
        dft['cap'] = 100
        dft['floor'] = 0
        with suppress_stdout_stderr():
            model.fit(dft)
        future = model.make_future_dataframe(periods=self.ndays_to_fcst,include_history=True)
        future['covid'] = future['ds'].apply(Covid_Range)
        future['cap'] = 100
        future['floor'] = 0
        pforecast = model.predict(future)
        forecast = pd.Series(pforecast['yhat'].values,index=pforecast['ds'].values,name='Occupancy Forecast')
        forecast = forecast.loc[forecast.index.isin(max_dates_to_fcst)]
        if self.ProdGraph:
            fig1 = model.plot(pforecast)
            fig2 = model.plot_components(pforecast)
            plt.show()
        return forecast
        
#--------------------------------------------------------------------------------------------------------------------

class ProphetFcst:
    '''
    previsione di occupancy per una 'Stay Date' futura basata sull'andamento avuto nel passato 
    (anno corrente ed anni precedenti) dell'occupancy di questa specifica data
    '''
    def __init__(self,row,last_book_day,last_occs,ProdGraph):
        self.row = row
        self.oday = row.name
        self.ndays_to_fcst = int((pd.to_datetime(self.oday)-(last_book_day+timedelta(days=1)))/np.timedelta64(1,'D'))+1
        self.dates_to_fcst = pd.date_range(start=last_book_day+timedelta(days=1), periods=self.ndays_to_fcst)
        self.df = row.reset_index()
        self.df.columns = ['ds','y']
        self.max_perc = 100
        if self.oday-pd.DateOffset(years=1) >= self.df['ds'].min():
            self.cap = self.df[self.df['ds']==self.oday-pd.DateOffset(years=1)]['y'].item()+self.max_perc
        else:
            self.cap = self.max_perc
        self.last_avail_occ = last_occs.loc[self.oday].item()
        self.ProdGraph = ProdGraph
    def GridSearchCV(self,param_grid,verbose):
        dfp = self.df
        grid = ParameterGrid(param_grid)
        train_days = len(dfp)-self.ndays_to_fcst-10
        bparams={}
        for params in grid:
            model=Prophet(**params)
            dfp['cap'] = self.cap
            with suppress_stdout_stderr():
                model.fit(dfp)
                df_cv = cross_validation(model, initial=str(train_days)+' days', period='1 days', horizon = str(self.ndays_to_fcst)+' days')
            df_p = performance_metrics(df_cv,rolling_window=1,metrics=['rmse'])
            bparams[(df_p['rmse'].values).item()] = params
        best_params = bparams[min(bparams)]
        if verbose==True:
            for rmse, par in sorted(bparams.items()):
                print(par,": %6.3f" %(rmse))
            print("\n","Best parameters:")
            print(best_params,'\n')
            exit(0)
        return best_params
    def CrossVal(self,params):
        err_type = CrossValPar[0]
        accuracy_thr = CrossValPar[1]
        quantile_val = CrossValPar[2]
        nfold = 100
        dfp = self.df
        horizon_days = (end_day-last_book_day).days
        period_days = 1
        initial_days = len(dfp)-(horizon_days+period_days*nfold) 
        model=Prophet(**params)
        dfp['cap'] = self.cap
        with suppress_stdout_stderr():
            model.fit(dfp)
            df_cv = cross_validation(model, initial=str(initial_days)+' days', period=str(period_days)+' days', horizon=str(horizon_days)+' days')
        if err_type == 'APE':
            err_df = pd.DataFrame({'horizon':df_cv['ds']-df_cv['cutoff'],'ape':abs((df_cv['yhat']-df_cv['y'])/df_cv['y'])})
            err_quant = err_df.groupby('horizon').quantile(quantile_val)
            err_quant.rename(columns={'ape':'mdape'},inplace=True)
        elif err_type == 'AE':
            err_df = pd.DataFrame({'horizon':df_cv['ds']-df_cv['cutoff'],'ae':abs((df_cv['yhat']-df_cv['y'])/100)})
            err_quant = err_df.groupby('horizon').quantile(quantile_val)
            err_quant.rename(columns={'ae':'mdae'},inplace=True)
        max_days_quant = int(np.nan_to_num(err_quant[err_quant.le(accuracy_thr)].dropna().index.days.max()))
        if max_days_quant < self.ndays_to_fcst:
            if max_days_quant == 0:
                eprint("!!! Hotel_id: ",fucsia_id,": numero di giorni nel futuro entro cui l'errore del forecast viene stimato sotto soglia pari a 0 !!!")
                exit(1)
            else:
                eprint('Quantile: ',quantile_val,'max ndays: ',max_days_quant,'lower than',self.ndays_to_fcst,'requested!')
        if err_type == 'APE':
            fig = plot_cross_validation_metric(df_cv, metric='mdape',rolling_window=0)
        elif err_type == 'AE':
            fig = plot_cross_validation_metric(df_cv, metric='mae',rolling_window=0)
        plt.savefig(dir+'img/'+str('cross_validation_performance_'+str(fucsia_id)))
        err_quant['quantile'] = quantile_val
        err_quant.to_csv(dirout_rms+"prediction_error_otbd_"+str(fucsia_id)+"_"+last_book_day.strftime('%Y-%m-%d')+".csv",header=True,index=True,sep=",",float_format='%.2f')
    def Forecast(self,params):
        holidays=pd.DataFrame(columns=['ds','holiday'])
        df_fcst = self.df
        df_fcst['covid'] = df_fcst['ds'].apply(Covid_Range)
        model=Prophet(interval_width=0.95,holidays=holidays,changepoint_range=0.9,**params)
        model.add_regressor('covid')
        model.add_country_holidays(country_name='IT')
        df_fcst['cap'] = self.cap
        with suppress_stdout_stderr():
            model.fit(df_fcst)
        future = model.make_future_dataframe(periods=self.ndays_to_fcst,include_history=True)
        future['covid'] = future['ds'].apply(Covid_Range)
        future['cap'] = self.cap
        pforecast = model.predict(future)
        model_fcst = pd.Series(pforecast['yhat'].values,index=pforecast['ds'].values,name=self.oday)
        model_fcst = model_fcst.where(model_fcst>0,0)
        model_fcst = np.maximum.accumulate(model_fcst[model_fcst.index > pd.to_datetime(self.oday)-pd.DateOffset(years=1)])
        last_past_fcst = model_fcst.loc[max(self.row.index)]
        delta = (model_fcst-last_past_fcst).loc[model_fcst.index.isin(self.dates_to_fcst)]
        forecast = (self.last_avail_occ+delta)
        forecast = forecast.where(forecast>=0,0)
        if self.ProdGraph:
            fig1 = model.plot(pforecast)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            fig1.text(0.2,0.9,str(self.oday),fontsize=16,verticalalignment='top', bbox=props)
            plt.savefig(dir+'img/'+str('prophet_fcst_'+self.oday.strftime('%Y-%m-%d')))
            fig2 = model.plot_components(pforecast)
            plt.savefig(dir+'img/'+str('prophet_components_'+self.oday.strftime('%Y-%m-%d')))
            plt.show()
        return forecast
    
#--------------------------------------------------------------------------------------------------------------------

def Get_PMS(fox2_id):
    # Get PMS historical data
    df = dd.read_csv("./PMS_ML.csv",sep=';',parse_dates=['StayDate','ValidityFrom','ValidityTo'],na_values=['2100-12-31']) # dask to reduce RAM usage
    df = df[df['fox2_id']==fox2_id].drop('fox2_id',axis=1)
    df = df[df['InventoryRMS']>0]
    df = df.compute()
    df['ValidityTo'].fillna(today,inplace=True)
    df['Date'] = df.apply(lambda row: pd.date_range(row['ValidityFrom'], row['ValidityTo']-relativedelta(days=1)), axis=1)
    df = df.explode('Date').drop(columns=['ValidityFrom','ValidityTo']).reset_index(drop=True)
    df.rename(columns={'StayDate':'oday','Date':'date','InventoryRMS':'inventory','RoomNights':'rooms'},inplace=True)
    df['occupancy'] = 100*df['rooms']/(df['inventory'])
    df.drop(['rooms','inventory'],axis=1,inplace=True)
    df.sort_values(by=['oday','date'],inplace=True)
    # considero solo lo storico entro 1 anno dalla data di prenotazione per avere una stagionalita su base annuale    
    df_date = df[(df['oday']<=end_day) & (df['oday']>=df['date']) & (df['oday'] <= df['date']+pd.DateOffset(years=1)-pd.DateOffset(days=1))]
    df_date = df_date[~df_date.duplicated(['oday','date'],keep='last')]
    df_oday = df[(df['oday'] <= last_book_day) & (df['oday']>=df['date'])]
    df_oday = df_oday[~df_oday.duplicated(['oday'],keep='last')] # ultimo dato
    df_oday.drop(['date'],axis=1,inplace=True)
    df_oday.loc[df_oday['occupancy']==0,'occupancy'] = np.nan
    return df_date, df_oday

#--------------------------------------------------------------------------------------------------------------------

def Get_CompDemIdx(fucsia_id):
    detail = dd.read_csv("./DatiHotel_ML.csv",sep=';',usecols=['idHotel','latitude','longitude','hotelStarsBookingCom'])
    detail = detail.dropna()
    detail.columns = ['fucsia_id','lat','long','stars']
    detail['stars'] = detail['stars'].replace(['RTA','NUL'], 0).astype(int)
    detail['lat'] = detail['lat'].astype(float)
    detail['long'] = detail['long'].astype(float)
    detail = detail.compute()
# brand reputation
    br = dd.read_csv("./Reputations_ML.csv",sep=';',usecols=['IdHotel','IdSourceShopper','ReviewName','ReviewValue','validityTo'],parse_dates=['validityTo'],na_values=['2100-12-31'])
    br = br[br['IdHotel'].isin(detail['fucsia_id'])]
    br['validityTo'] = br['validityTo'].fillna(today)
    br = br[br['IdSourceShopper']==1]
    br = br.compute()
    br.sort_values(by=['IdHotel','ReviewName','validityTo'],inplace=True)
    br = br[~br.duplicated(['IdHotel','ReviewName'],keep='last')]
    br.drop(['IdSourceShopper','validityTo'],axis=1,inplace=True)
    br.set_index('IdHotel',inplace=True)
    br = br.pivot(columns='ReviewName')
    br.index.name = 'fucsia_id'
    br.columns = br.columns.droplevel(0)    
    br.columns.name = ''
    br.rename(columns={'HOTEL_REVIEW':'nrev','HOTEL_TOTAL':'br_tot','HOTEL_SERVICES':'br_services','HOTEL_CLEAN':'br_clean',
                       'HOTEL_COMFORT':'br_comfort','HOTEL_VALUE':'br_value','HOTEL_LOCATION':'br_location','HOTEL_WIFI':'br_wifi'},inplace=True)
    br = br[br['nrev']>0]
    if cfg_ver>=5:
        br = br[br['br_value']>=8]
    br.dropna(inplace=True)
    df_feat = pd.merge(detail.set_index(['fucsia_id']),br,left_index=True,right_index=True,how='inner')
    df_feat.dropna(subset=features,inplace=True)
# PCA
    if PCAtrans:
        br_features = list(filter(lambda x: x.startswith('br_'), features))
        not_br_features = list(filter(lambda x: not x.startswith('br_'), features))
        df_feat_bar = pd.DataFrame(data=StandardScaler().fit_transform(df_feat[br_features]),index=df_feat.index,columns=df_feat[br_features].columns)
        pca = PCA(n_components=0.95)
        pca.fit(df_feat_bar)
        df_feat_bar = pd.DataFrame(data=pca.transform(df_feat_bar),index=df_feat_bar.index)
        df_feat_bar = df_feat_bar.add_prefix('br_')    
        df_feat_bar = pd.concat([df_feat[not_br_features],df_feat_bar],axis=1)
    else:
        df_feat_bar = df_feat[features]
    features_bar = df_feat_bar.columns

    feat_master = df_feat_bar.loc[df_feat_bar.index == fucsia_id]
    DataMiss=False
    if len(feat_master) == 0: 
        eprint('!!! fucsia_id: '+str(fucsia_id)+' cfg: '+str(cfg_ver)+' not found in features data !!!')
        DataMiss=True
# rates
    bar = dd.read_csv("./Rates_ML.csv",sep=';',parse_dates=['StayDate'])
    bar = bar[bar['idFucsiaShopper'].isin(detail['fucsia_id'])]
    bar = bar[bar['StayDate']>=first_fcst_day]
    bar = bar.compute()
    bar.rename(columns={'idFucsiaShopper':'fucsia_id','StayDate':'rday'},inplace=True)
    bar = bar[~bar.duplicated(['fucsia_id','idSourceShopper','rday'],keep='last')]
    bar['minrate'] = bar['minrate'].replace([0,-1], np.nan)
    bar.set_index(['fucsia_id','rday'],inplace=True)
    bar = bar.pivot(columns='idSourceShopper')
    bar.columns = bar.columns.droplevel(0)
    bar.columns.name = ''
    if RateChannel == 'min':
        bar['BAR'] = bar[[1,2]].min(skipna=True,axis=1).drop([1,2],axis=1)
    elif RateChannel == 'bcom':
        bar['BAR'] = bar[1]
    elif RateChannel == 'hcom':
        bar['BAR'] = bar[2]
    bar.drop([1,2],axis=1,inplace=True)
    avg_bar = bar.groupby(level='fucsia_id')['BAR'].mean()
    bar = bar[bar.index.get_level_values('rday').isin(max_dates_to_fcst)]
    rates = pd.DataFrame(index=pd.date_range(first_fcst_day, end_day),columns=['bar','cap','floor'])
    if fucsia_id in bar.index:
        rates['bar'] = bar['BAR'].loc[fucsia_id]
    else:
        eprint('!!! fucsia_id: '+str(fucsia_id)+' cfg: '+str(cfg_ver)+' not found in rates data !!!')
        DataMiss=True
    if DataMiss:
        exit(1)
# identify outliers
    dfall = pd.merge(df_feat_bar,bar[bar['BAR'].notna()]['BAR'].reset_index(),left_index=True,right_on='fucsia_id',how='inner').set_index(['fucsia_id','rday'])
    outlier = pd.Series(index=dfall.index,name='outlier')
    df_idx = pd.DataFrame(index=pd.date_range(start=first_fcst_day, end=end_day),columns=['comp_idx','comp_idx_std','market_rate','demand_idx'])
    for day,df in dfall.groupby('rday'):
        X = df.drop(['BAR'],axis=1)
        regr_bar = RandomForestRegressor(**params_bar).fit(X, df['BAR'])   
        market_rate = regr_bar.predict(X)
        comp_idx = df['BAR']/market_rate
        median = comp_idx.quantile(0.5)
        mad = stats.median_abs_deviation(comp_idx,scale='normal')
        upper_limit = median+3*mad
        lower_limit = median-3*mad
        outlier.loc[comp_idx.index] = ~comp_idx.between(lower_limit,upper_limit)
        df_idx['comp_idx_std'].loc[day] = comp_idx.mask(outlier==True).std()
        if fucsia_id in X.index:
            rates.loc[day,'cap'] = RateChangePar[0]*regr_bar.predict(X.loc[fucsia_id])*upper_limit
            rates.loc[day,'floor'] = RateChangePar[1]*regr_bar.predict(X.loc[fucsia_id])*lower_limit
    if fucsia_id in outlier.index:
        rates.mask(outlier.loc[fucsia_id]==True,inplace=True) # remove outlier rates
    rates.index = rates.index.strftime('%Y-%m-%d')
    bar.mask(outlier==True,inplace=True) # remove outlier rates
    df_avg = pd.merge(df_feat_bar,avg_bar,left_index=True,right_index=True,how='inner')
    avg_all = df_avg.loc[df_avg.index != fucsia_id]
    dfall = pd.merge(df_feat_bar,bar.reset_index(level='rday'),left_index=True,right_index=True,how='inner')
    frac_avail_day = dfall.groupby(dfall.index)['BAR'].count()/next_days_to_fcst # frazione di giorni in cui gli hotels sono disponibili
    hf_avail_hotels = frac_avail_day[frac_avail_day >= MarketRatePar[0]].index # hotels disponibili per una frazione di giorni sopra la soglia MarketRatePar[0]
    avg_avail = avg_all[(avg_all['BAR'].notna()) & (avg_all.index.isin(hf_avail_hotels))]
    BestParams = False
    if BestParams:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        regr = RandomForestRegressor(random_state=0)
        feat = features_bar
        X = avg_avail[feat]
        y = avg_avail['BAR']
        param_list = {'n_estimators': [100],'max_features': ['sqrt'],'max_depth' : [13,14,15],'criterion' :['mae'],'min_samples_leaf':[3]}
        #grid_search = HalvingGridSearchCV(regr, param_list, factor=2, cv=5, random_state=0, n_jobs=-1).fit(X, y)
        grid_search = GridSearchCV(regr,param_list,cv=10,n_jobs=-1).fit(X, y)
        best_model = grid_search.best_estimator_
        print(grid_search.best_params_, grid_search.best_score_)
        cvres = pd.DataFrame(grid_search.cv_results_)
        #print(cvres[['iter','mean_test_score','params']])
        print(cvres[['mean_test_score','params']].sort_values(by=['mean_test_score'],ascending=False))
        #model = RandomForestRegressor(**params_bar).fit(X, y)
        feat_imp = pd.Series(best_model.feature_importances_, list(X)).sort_values(ascending=False)
        print(feat_imp)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()
        plt.savefig('Features_importance')
        plt.show()
        exit(0)

    regr_bar = RandomForestRegressor(**params_bar).fit(avg_avail[features_bar], avg_avail['BAR'])
    avg_market_rate = regr_bar.predict(feat_master[features_bar]).item()
    for day in dfall.loc[dfall.index == fucsia_id]['rday']:
        dfday = dfall[dfall['rday']==day].drop(['rday'],axis=1)
        master = dfday.loc[dfday.index == fucsia_id]
        all_sample = dfday.loc[dfday.index != fucsia_id]        
        avail_sample = all_sample[(all_sample['BAR'].notna()) & (all_sample.index.isin(hf_avail_hotels))]
        regr_bar = RandomForestRegressor(**params_bar).fit(avail_sample[features_bar], avail_sample['BAR'])   
        market_rate = regr_bar.predict(master[features_bar]).item()
        comp_idx = master['BAR'].item()/market_rate
        demand_idx = (market_rate/avg_market_rate)
        df_idx.loc[day][['comp_idx','market_rate','demand_idx']] = [comp_idx,market_rate,demand_idx]
    df_idx.index = df_idx.index.strftime('%Y-%m-%d')
    npoints = df_idx['market_rate'].count()
    if MarketRatePar[1] and npoints >= 30:
        # Fitto i dati del market rate con prophet per ottenere la loro migliore rappresentazione (al netto delle fluttuazioni casuali)
        market_rate_raw = df_idx['market_rate']                
        dfp = df_idx['market_rate'].reset_index()
        dfp.columns = ['ds', 'y']
        holidays=pd.DataFrame(columns=['ds','holiday'])
        if npoints >= 60:
            params = {'weekly_seasonality': 1,
                      'changepoint_prior_scale':1,
                      'holidays_prior_scale':1,
                      'seasonality_prior_scale': 1}
        elif npoints >= 30:
            params = {'weekly_seasonality': 1,
                      'changepoint_prior_scale':1,
                      'n_changepoints':20,
                      'holidays_prior_scale':1,
                      'seasonality_prior_scale': 1}            
        model=Prophet(interval_width=0.683,changepoint_range=1,holidays=holidays,**params)
        model.add_country_holidays(country_name='IT')
        with suppress_stdout_stderr():
            model.fit(dfp)
        future = model.make_future_dataframe(periods=0)
        pforecast = model.predict(future)
        df_idx['market_rate'] = pforecast['yhat'].values
        df_idx['comp_idx'] = df_idx['comp_idx']*market_rate_raw/df_idx['market_rate']
        df_idx['demand_idx'] = df_idx['demand_idx']*df_idx['market_rate']/market_rate_raw
    return rates, df_idx

#--------------------------------------------------------------------------------------------------------------------

def date_of_stay_cxr(x):
    drange = pd.date_range(start=x['check-in'], end=x['check-out']-relativedelta(days=1)).strftime('%Y-%m-%d')
    if x['Stato'] == 'Cancel':
        x['Stato'] = 0
    else:
        x['Stato'] = 1
    return dict({'stato':x['Stato'],'canale':x['Operatore'],'date':x['date'],'oday':drange,'los':len(drange)-1,'bw':(x['check-in']-x['date']).days})

#--------------------------------------------------------------------------------------------------------------------

def date_of_stay(x):
    drange = pd.date_range(start=x['check-in'], end=x['check-out']-relativedelta(days=1)).strftime('%Y-%m-%d')
    return dict({'canale':x['Operatore'],'date':x['date'],'oday':drange})

#--------------------------------------------------------------------------------------------------------------------

def same_weekday_prev_years(x):
# modifico le serie temporali degli anni precedenti in modo da riferirsi allo stesso giorno della settimana della previsione
    spec_day_mask = np.array(['08-15','12-25','12-31'])
    min_day = x['oday'].min()
    for fday in max_dates_to_fcst:
        nyears = relativedelta(fday,min_day).years
        SDay = False
        if np.isin(fday.strftime('%m-%d'),spec_day_mask):
            SDay = True
        xn = pd.DataFrame()
        for ny in range(nyears+1):
            if ny > 0:
                if SDay:
                    fday_py = fday-pd.DateOffset(years=ny)
                    ndays = timedelta(days=0)
                else:
                    fday_py = fday-pd.to_timedelta(round((fday-(fday-pd.DateOffset(years=ny))).days/7,0),'W')
                    ndays = pd.to_timedelta((pd.to_datetime(fday)-pd.DateOffset(years=ny))-fday_py, unit='d')
                x['date'] = np.where(x['oday']==fday_py.strftime('%Y-%m-%d'),x['date']+ndays,x['date'])
                x['oday'] = np.where(x['oday']==fday_py.strftime('%Y-%m-%d'),pd.Series(fday),x['oday'])
    x = x[x['oday'].isin(max_dates_to_fcst)] # seleziono solo i giorni per il forecast
    x.set_index(['date'],inplace=True)
    return x

#--------------------------------------------------------------------------------------------------------------------

def avg_net_occ(x):
# estraggo il trend monotonicamente crescente dalla time series ed a questo applico la cancellazione media    
    df = x.set_index('date').transform(lambda y: np.where(~y.diff().isnull(),y.diff(),y))
    net_occ = df[df>=0].join(df[df<0], lsuffix="_A", rsuffix="_B").cumsum().fillna(method='ffill').fillna(0)
    net_occ.columns = ['pos','neg']
    if net_occ['pos'][-1]:
        cxr = 1-abs(net_occ['neg'][-1]/net_occ['pos'][-1])
    else:
        cxr = 1
    net_occ['pos'] = net_occ['pos']*cxr
    net_occ = net_occ.drop('neg',axis=1)
    return net_occ

#--------------------------------------------------------------------------------------------------------------------

def occ_smoothing(x,wind_smooth):
    xs = x.drop(['date'],axis=1)
    smoothed=xs.rolling(window=wind_smooth, win_type='gaussian', center=True).mean(std=7)
    smoothed.iloc[0]=xs.iloc[0]
    smoothed.iloc[-1]=xs.iloc[-1]
    smoothed=smoothed.interpolate(method='akima')
    return smoothed
    
#--------------------------------------------------------------------------------------------------------------------

def interpolate(df,yesterday,source):
#  riempio i valori delle date mancanti con quelli della prima data precedente (per ogni canale e per ogni ciclo annuale)
    oday = df.name
    df = df.reset_index()
    df['nyear'] = df.apply(lambda x: relativedelta(pd.to_datetime(oday), x['date']).years, axis=1)
    if source == 'pms':
        idx = ['nyear']
        idx2 = ['date']
    elif source == 'crs':
        idx = ['canale','nyear']
        idx2 = ['canale','date']
    df[oday] = df.groupby(idx)[oday].ffill().fillna(0)
    last_book_day = min(pd.to_datetime(oday),yesterday)
    df[oday] = np.where(df['date'] > last_book_day, np.nan, df[oday])
    df.drop('nyear',axis=1,inplace=True)
    df.set_index(idx2,inplace=True)
    return df.squeeze()

#--------------------------------------------------------------------------------------------------------------------

def hierarchical_rate_avg(x,thr,stars_avail,star):
    x = x.droplevel('rday')
    x = x.reindex(np.unique(np.append(stars_avail,[4,5])))
    x.fillna(0,inplace=True)
    if x.loc[star]['count'] >= thr:
        havg = x.loc[star]['mean']
    else:
        nt = x.loc[star]['count']
        rt = x.loc[star]['mean']*x.loc[star]['count']
        idx = np.flip(x.index.values)
        if star < 4:
            midx = list(idx).index(star)
            idx = np.roll(idx,-midx)
            idx = np.append(np.delete(idx,[list(idx).index(4),list(idx).index(5)]),[4,5])
        idx = np.delete(idx,list(idx).index(star))
        for i in idx:
            if nt < thr:
                nt += x.loc[i]['count']
                rt += x.loc[i]['mean']*x.loc[i]['count']
        havg = rt/nt
    return havg

#--------------------------------------------------------------------------------------------------------------------

def occ_max_range(nu,rate0,occupancy_otb,occupancy_fcst,gravity,rate_disp_width,occupancy_max_thr,getpar):
    q = np.exp(((gravity-rate0)*np.log((nu*np.sqrt(nu**2+6*nu+5)+nu**2+3*nu)/2)-(gravity+rate_disp_width-rate0)*np.log(nu))/(-rate_disp_width))
    occupancy_max = occupancy_otb+(occupancy_fcst-occupancy_otb)*(1+q)**(1/nu)
    if gravity != rate0:
        el = np.log(nu/q)/(gravity-rate0)
    else:
        el = np.log((nu*np.sqrt(nu**2+6*nu+5)+nu**2+3*nu)/(2*q))/(gravity+rate_disp_width-rate0)
    if getpar:
        return occupancy_max,q,el
    else:
        return abs(occupancy_max_thr-occupancy_max)

def generalised_logistic_demand(rate,rate0,occupancy_otb,occupancy_fcst,gravity,rate_disp_width,capacity,occupancy_max_thr):
    """ Find roots of a system of equations to get parameter values of the occupancy modeled as a generalised logistic function of rate. 
    Returns -RevPar to be minimized in order to find rate corresponding to maximum RevPar"""
    if occupancy_fcst > occupancy_otb and occupancy_fcst < occupancy_max_thr:
        opt_res = optimize.minimize_scalar(occ_max_range, args=(rate0,occupancy_otb,occupancy_fcst,gravity,rate_disp_width,occupancy_max_thr,False), bounds=[1e-14,100], method='bounded')
        nu = opt_res['x']
        occupancy_max, q, el = occ_max_range(nu,rate0,occupancy_otb,occupancy_fcst,gravity,rate_disp_width,occupancy_max_thr,getpar=True)
        occupancy = occupancy_otb+(occupancy_max-occupancy_otb)/(1+q*np.exp(el*(rate-rate0)))**(1/nu)
        occupancy[occupancy > capacity] = capacity
        add_occupancy = occupancy-occupancy_otb
        revpar = rate*add_occupancy        
    else:
        revpar = 0
    return -revpar

#--------------------------------------------------------------------------------------------------------------------

def logistic_demand(rate, occupancy_fcst, elasticity, rate0, capacity, occupancy_otb):
    """Returns -RevPar, given other parameter values, modeling occupancy as a logistic function of rate. 
    Minimized to find rate corresponding to maximum RevPar"""
    if occupancy_fcst > occupancy_otb and occupancy_fcst < capacity:
        q = (capacity-occupancy_fcst)/(occupancy_fcst-occupancy_otb)
        occupancy = occupancy_otb+(capacity-occupancy_otb)/(1+q*np.exp(elasticity*(rate-rate0)))
    elif occupancy_fcst >= capacity:
        occupancy = capacity
    else:
        occupancy = occupancy_otb
    add_occupancy = occupancy-occupancy_otb
    revpar = rate*add_occupancy
    return -revpar

#--------------------------------------------------------------------------------------------------------------------

def eprint(*args, **kwargs):
    # write to stderr
    print(*args, file=sys.stderr, **kwargs)
