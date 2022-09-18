from scipy.optimize import minimize_scalar
from scipy import integrate
import pandas as pd
import csv
import pickle
#import pickle5
from scipy.linalg import eigh, cholesky
from scipy.stats import norm, johnsonsu,spearmanr,skew, kurtosis
import ipywidgets as wid
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import apimoex
import requests
from pandas_datareader import data as pdr
from datetime import datetime as dt
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fsolve
import matplotlib.ticker as mtick
import sys
import SPmodule as sp
import quandl

class bond():
    spec_attr=['ISUSD','ISEUR','coupon_rate','thresholds']

    def __init__(self):
        self.bond_name='' # Название облигации
        self.ISIN='' # ISIN
        self.tickers=[] # список названия базовых активов
        self.data_source='yahoo' # Источник цен
        self.dateformat=None  # 
        self.issue_date='' # Дата выпуска облигации
        self.analysis_date='' # Дата проведения анализа 
        self.coupon_dates=[''] #Даты купнов 
        self.start_observation = '' # начало наблюдений 
        self.observations = [''] # Даты наблюдений 
        self.base_source = '' # Страна календаря ?
        self.include_dividends=False # Флаг включения дивидендов
        self.fix_cf=[] # Фиксированные потоки 
        self.dirty_price = 0 # грязная цена облигации на дату анализа = рыночная цена + НКД
        self.YTM_method = '' #- выбор способа расчета YTM по справедливой цене или по грязной
        self.principal=0 # Номинал облигации
        self.risk_spread = 0 # Показатель кредитного спреда 
        self.ISUSD=False # Крррекция на курс доллара  
        self.ISEUR=False #  Крррекция на курс евро
        self.usd=1 # 
        self.eur=1 #
        self.price_calc_type=None #
        self.cap=np.inf #максимальный доход в процентах
        self.CDS=[] # Ток 4й тип
        self.libor=[] # Ставка Либор Ток 4й тип 
        self.vol_days=[] # Ток 4й тип
        
   
    def calc_fair_price(self):
        '''
        Расчет стоимости фиксированной части, опциона и справедливой стоимости облигации после Монте-Карло
        '''
        self.opt_price = self.calc_price_exp(r=np.asarray(self.base_rate), cfs=self.opt_payoff, yrs=self.dates_years[1:]) # стоимость опциона
        self.opt_price_db = dict(enumerate(self.opt_price.flatten(), 1))
        
        if self.Base_Curr == 'RUB':
            self.nominal_price = self.calc_price(r=np.asarray(self.rf_rate)+self.risk_spread, cfs=self.fix_cf, yrs=self.dates_years[1:]) # стоимость облигации
        else:
            self.nominal_price = self.calc_price_exp(r=np.asarray(self.base_rate+self.risk_spread), cfs=self.fix_cf, yrs=self.dates_years[1:])
        self.nominal_price_db = dict(enumerate(self.nominal_price.flatten(), 1))
        self.fair_price = self.opt_price + self.nominal_price #справедливая стоимость
        self.fair_db = dict(enumerate(self.fair_price.flatten(), 1))
        
    
    #Расчет дисконтированной стоимости по массиву ставок, разным для каждого денежного потока
    def calc_price(self,r, cfs, yrs):
        return np.sum(np.asarray(cfs)/(1. + np.asarray(r)) ** np.asarray(yrs))
   
    def fair_price_distr(self): # расчёт персентилей для справедливой цены
        self.f_prices_matrix = [self.f_prices[i:i + len(self.cf_sim)] for i in range(0, len(self.f_prices), len(self.cf_sim))]
        res_prices = []                                     # определение ст-ти облигации при каждом сценарии
        for cf in self.f_prices_matrix:
            res_prices.append(self.calc_price_exp(r=np.asarray(self.base_rate), cfs = cf, yrs=self.dates_years[1:]))
        self.fair=pd.DataFrame(data=np.round(np.percentile(res_prices,np.linspace(100,0,101)),6),columns=['Fair_Prices'], index=[i/100 for i in range(101)]) 
        self.fair['Median Fair price']=np.median(res_prices)
        self.fair['Mean Fair price']=np.mean(res_prices)
        self.interval = (self.nominal_price+self.opt_price - 1.96*(np.std(res_prices)/np.sqrt(self.n_windows*self.n_scenarios)),
                         self.nominal_price + self.opt_price + 1.96*(np.std(res_prices)/np.sqrt(self.n_windows*self.n_scenarios)))
        self.fair.index.name="FV percentiles for "+self.bond_name
     #Расчет дисконтированной стоимости по массиву ставок, разным для каждого денежного потока   
    
    def calc_price_exp(self,r, cfs, yrs):
        return np.sum(np.asarray(cfs)*np.exp(-np.asarray(r)* np.asarray(yrs)))
        
class bond_1(bond):
    def __init__(self):
        super().__init__()
        self.init_not_paid=0 #сколько раз не платили 
        self.am_barier_triggered=False
        self.end_payment = False
        self.usd=1
        self.eur=1
        self.bond_type=1
        self.thresholds=[] #барьер
        self.coupon_rate=0 #Размер купона 
        self.memory_K=[] # Коэфф-т
        self.american_barrier=False
        
    def hist(self, rets, first_date, last_date):
        if first_date>last_date:
            first_date=last_date
        self.init_not_paid=0 #сколько раз не платили 
        self.am_barier_triggered=False
        self.usd=1
        self.eur=1
        self.cfs=[0 for j in range(len(self.coupon_dates))]
        self.hist_prices = [1 for i in range(len(self.tickers))]
        
        #print('rets \n', rets.loc[first_date:last_date, self.tickers].cumsum(axis=0))
        #print('rets2 \n', rets.loc[first_date, self.tickers])
        self.massiv = self.hist_prices * np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        print('Calc prices\n', self.massiv.loc[last_date,self.tickers]*self.prices.loc[first_date,self.tickers].values)
        print('Fact prices\n', self.prices.loc[last_date,self.tickers])
        if self.ISUSD == True:
            self.usd_values = self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR == True:
            self.eur_values = self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])    
        self.calc(rets, first_date, last_date)
        self.hist_prices = self.massiv.loc[last_date,self.tickers]
        if self.ISUSD == True: self.usd=self.usd_values.loc[last_date]
        if self.ISEUR == True: self.eur=self.eur_values.loc[last_date]
        self.init_not_paid = self.not_paid
        print('init_not_paid', self.init_not_paid)
        if self.american_option == True: 
            if sum(self.massiv.loc[:, self.tickers].min() <= self.american_barrier) == len(self.tickers):
                self.am_barier_triggered=True
                print("self.am_barier_triggered",self.am_barier_triggered)
        print('hist_prices:\n', self.hist_prices)
        if self.ISUSD==True: print('usd',self.usd)
        if self.ISEUR==True: print('eur',self.eur)
        
    def calc(self, rets, first_date, last_date):
        self.massiv = self.hist_prices * np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0) - rets.loc[first_date, self.tickers])
        #print(self.massiv.loc[:self.start_obs,self.tickers])
        if self.ansys_date <= self.start_obs and last_date >= self.start_obs:
            self.massiv.loc[:, self.tickers] /= self.massiv.loc[self.start_obs, self.tickers]
        #print(self.massiv.loc[:self.start_obs,self.tickers])
        
        #print(self.massiv.loc[:,self.tickers]*self.prices.loc[self.start_obs,self.tickers].values)
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])
        self.not_paid=self.init_not_paid
                
        for j in range(len(self.observ_dates)):
            #print('date',self.observ_dates[j])
            #проверяем, что даты наблюдения входят в диапазон
            if (first_date<self.observ_dates[j]) and (self.observ_dates[j]<=last_date):            
                #print(self.massiv.loc[self.observ_dates[j], self.tickers])
                #проверяем, что количество БА, преодолевших барьер на дату наблюдения, равно количеству тикеров (то есть все активы выше барьера)
                if sum(self.massiv.loc[self.observ_dates[j], self.tickers]>self.thresholds[j])==len(self.tickers):
                    #print(self.massiv.loc[self.observ_dates[j], self.tickers])
                    self.cfs[j] = self.coupon_rate*(self.not_paid+self.memory_K[j])*self.principal
                    #print('cfs bef USD',self.cfs)
                    self.not_paid = 0
                else:
                    self.cfs[j] =0
                    #print('cfs zero',self.cfs)
                    self.not_paid += self.memory_K[j]# @ изменено
                if self.ISUSD==True:
                    #print('USD mult ',self.usd_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    self.cfs[j]*=self.usd_values.loc[self.observ_dates[j]]
                    #print('cfs after USD',self.cfs)
                elif self.ISEUR==True:
                    #print('EUR mult ',self.eur_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                    self.cfs[j]*=self.eur_values.loc[self.observ_dates[j]]
                    #print('cfs after EUR',self.cfs)
                
                #если расчет по american option и последняя дата
                if self.observ_dates[j]==self.observ_dates[-1] and self.american_option == True:
                    #если цены ба выше первоначальных или если цены ба не касались и не снижались ниже барьера на всем периоде
                    #print('American option, last date, prices must be more than 1 \n', self.massiv.loc[self.observ_dates[j], self.tickers])
                    #print('Minimum: ', self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].min())
                    #print('American barrier: ', self.american_barrier)
                    if sum(self.massiv.loc[:, self.tickers].min() <= self.american_barrier)==len(self.tickers):
                        self.am_barier_triggered=True
                    if (sum(self.massiv.loc[self.observ_dates[j], self.tickers] >= 1) == len(self.tickers)) \
                        and self.am_barier_triggered==False:
                        #print('American option, last date, prices must be more than 1 \n', self.massiv.loc[self.observ_dates[j], self.tickers])
                        #print('Minimum: ', self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].min())
                        #print('Последний поток: ',self.cfs[-1])
                        self.cfs[-1] += self.principal
                        #print('После увеличения: ', self.cfs[-1])
                    #если цены ниже первоначальных
                    else:
                        #определяем цену худшего
                        self.price_sort = self.massiv.loc[self.observ_dates[j], self.tickers].sort_values()[0]
                        #print('Minimum else: ', self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].min())
                        #print('Minimum else self.price_sort: ',self.price_sort)
                        #последний поток равен 
                        self.cfs[-1] += np.maximum(self.protection, self.price_sort)*self.principal
                        #print('self.cfs[-1]: ',self.cfs[-1])
            if self.observ_dates[j] == self.observ_dates[-1] and self.end_payment == True:
                a = sum(self.cfs)
                self.cfs = [0 for j in range(len(self.coupon_dates))]
                self.cfs[-1] = a
                
        return self.cfs   
     

class bond_2(bond):
    spec_attr=['ISUSD','ISEUR','participation_K','price_calc_type', 'basket_perf']  
    
    def __init__(self):
        super().__init__()
        self.hist_prices = [1 for i in range(len(self.tickers))]
        
        self.usd=1
        self.eur=1
        self.average_sum=0
        self.average_count=0
        self.init_average_sum=0
        self.init_average_count=0 
        self.bond_type = 2
        self.participation_K = 0 #Коэфф участия
        self.price=0
    
    def hist(self, rets, first_date, last_date):
        if first_date>last_date:
            first_date=last_date
        self.hist_prices = [1 for i in range(len(self.tickers))]
        self.am_barier_triggered=False
        self.usd=1
        self.eur=1
        self.average_sum=0
        self.average_count=0
        self.init_average_sum=0
        self.init_average_count=0 
        self.cfs=[0 for j in range(len(self.coupon_dates))]
        print(self.prices)        
        self.massiv = self.hist_prices*np.exp(rets.loc[first_date: last_date,self.tickers].cumsum(axis=0) - rets.loc[first_date,self.tickers])
                
        print(self.massiv.loc)
        print('Calc prices\n', self.massiv.loc[last_date,self.tickers]*self.prices.loc[first_date,self.tickers].values)
        print('Fact prices\n', self.prices.loc[last_date,self.tickers])
        
        if self.basket_perf == True and all(x == 1 for x in self.basket_init_price)==False:
            print('Initial prices\n', self.basket_init_price)
            self.basket_init_price = self.basket_init_price / self.prices.loc[last_date,self.tickers] 
            print('Initial prices\n', self.basket_init_price)
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])
        
        self.calc(rets,first_date, last_date)
        self.hist_prices=self.massiv.loc[last_date,self.tickers]
        
        self.init_waterline = self.waterline
        
        if self.ISUSD==True: self.usd=self.usd_values.loc[last_date]
        if self.ISEUR==True: self.eur=self.eur_values.loc[last_date]
        if (self.price_calc_type == 'american_min') and (self.massiv.loc[:, self.tickers].min().values[0] <= self.american_barrier):
            self.am_barier_triggered=True
            print("self.am_barier_triggered",self.am_barier_triggered)
        if (self.price_calc_type == 'american_max') and (self.massiv.loc[:, self.tickers].max().values[0]>= self.american_barrier):
            self.am_barier_triggered=True
            print("self.am_barier_triggered",self.am_barier_triggered)
        print('init_waterline:\n', self.init_waterline)
        print('hist_prices: ',self.hist_prices)
        if self.ISUSD==True: print('usd',self.usd)
        if self.ISEUR==True: print('eur',self.eur)
        
        self.init_average_sum = self.average_sum
        self.init_average_count = self.average_count
        print("init_average_sum", self.init_average_sum)
        print("init_average_count", self.init_average_count)
        
    def calc(self, rets, first_date, last_date):        
        
        self.massiv= self.hist_prices*np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        #print(self.massiv)
        #print('Calc prices\n', self.massiv.loc[last_date,self.tickers]*self.prices.loc[first_date,self.tickers].values)
        if self.ansys_date <= self.start_obs and last_date >= self.start_obs:
            self.init_waterline = self.massiv.loc[self.start_obs, self.tickers].values[0]
        self.waterline = self.init_waterline
        
        if self.price_calc_type == 'average':
            self.average_sum = self.init_average_sum
            self.average_count = self.init_average_count
            self.fbm_values=[]
            self.first_dm = pd.date_range(first_date, last_date, freq='MS')
            
            #print(self.average_sum,self.average_count)
            
            for i in self.first_dm:
                self.fbm_values.append(self.massiv.loc[i:(i+timedelta(31)), self.tickers].iloc[0].item())
                #print(self.massiv.loc[i:(i+timedelta(31)), self.tickers].index[0])
            #print(sum(fbm_values),len(fbm_values))
            
            self.average_sum += sum(self.fbm_values)
            self.average_count += len(self.fbm_values)
            
            #print(self.average_sum,self.average_count)
        
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date,'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values = self.eur*np.exp(rets.loc[first_date:last_date,'EURRUB'].cumsum(axis=0)-rets.loc[first_date,'EURRUB'])
            
        for j in range(len(self.observ_dates)):
            #print('date',self.observ_dates[j])
            
            if (first_date < self.observ_dates[j]) and (self.observ_dates[j] <= last_date): 
                self.cfs[j]=0
                if self.price_calc_type == 'every':
                    self.price = self.massiv.loc[self.observ_dates[j],self.tickers].values[0]
                    self.cfs[j] = (min(self.cap, self.participation_K * max(0, self.price-self.waterline))) * self.principal
                    self.waterline = max(self.massiv.loc[self.observ_dates[j], self.tickers].values[0], self.waterline)
                    
                elif self.observ_dates[j] == self.observ_dates[-1]:
                    if self.price_calc_type == 'end':
                        if self.basket_perf == False:
                            self.price = self.massiv.loc[self.observ_dates[j],self.tickers].values[0]
                        else:
                            #print('Числитель', self.massiv.loc[self.observ_dates[j],self.tickers].values)
                            #print('Знаменатель', self.basket_init_price)
                            self.price = self.massiv.loc[self.observ_dates[j],self.tickers] / self.basket_init_price
                            #print('Отношение', self.price)
                            self.price=self.price.mean()
                            #print('Среднее', self.price)

                    elif self.price_calc_type == 'average':
                        self.price = self.average_sum / self.average_count
                    #print('ret',price)
                    
                    if self.price_calc_type == 'american_min':
                        #если цена на последнюю дату выше начальной
                        self.price = self.massiv.loc[self.observ_dates[j],self.tickers].values[0]
                        #print('SelfPrice: ',self.price)
                        #если она не превышает начальную, 
                        if self.price < 1:
                            #то берем ее отклонение в другую сторону
                            self.price = 2 - self.price
                            #print('SelfPrice < 1, тогда 2-self.price = ',self.price)
                            #если за весь период минимальное значение цены было меньшеравно барьера
                            if (self.am_barier_triggered==True) or (self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].min().values[0] <= self.american_barrier):
                                #print('Минимум цены за весь период: ', self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].min().values[0])
                                #np.minimum(self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].values[0])
                                self.price = (1 + self.american_coupon)
                                #print('Барьер пробивался', self.price)
                    
                    if self.price_calc_type == 'american_max':
                        #если цена за весь период не превышала барьер,
                        #print('Максимум цены за весь период: ', self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].max().values[0])
                        self.max = self.massiv.loc[self.observ_dates[0]:self.observ_dates[-1], self.tickers].max().values[0]
                        #print('check self.max: ',self.max)
                        if (self.am_barier_triggered==True) or (self.max >= self.american_barrier):
                            #то выплачиваем купон
                            self.price = (1 + self.american_coupon)
                            #print('SelfPrice with coupon: ',self.price)
                        else:
                            self.price = self.massiv.loc[self.observ_dates[j],self.tickers].values[0]
                            #print('SelfPrice without coupon: ',self.price)
                    
                    #расчет основного потока
                    if (self.observ_dates[j] == self.observ_dates[-1]) & (self.commission != 0):
                        #print('Цена БА в последний день =', self.price)
                        if (self.massiv.loc[self.observ_dates[j],self.tickers].values[0] >= 1):
                            self.princip_with_commiss = self.principal * (1 - self.commission * self.days_between / 365)
                            #print('waterline =', self.waterline)
                            self.cfs[j] = (min(self.cap,self.participation_K * max(self.floor,self.price))) * self.princip_with_commiss
                            #print('Цена погашения с учетом комиссии =',self.princip_with_commiss)
                            #print('Денежный поток =',self.cfs[j])
                        elif (self.massiv.loc[self.observ_dates[j],self.tickers].values[0] < 1):
                            # можно опустить действие и сразу в формуле основного потока домножить на номинал и комиссию
                            self.princip_with_commiss = self.principal * (self.massiv.loc[self.observ_dates[j],self.tickers].values[0] - self.commission * self.days_between / 365)
                            self.cfs[j] = (min(self.cap,self.participation_K * max(self.floor,self.price))) * self.princip_with_commiss
                            #print('Цена погашения с учетом комиссии =',self.princip_with_commiss)
                            #print('Денежный поток =',self.cfs[j])
                    else:
                        self.cfs[j] = (min(self.cap,self.participation_K * max(self.floor,self.price - self.waterline))) * self.principal
                        #print('Денежный поток =',self.cfs[j])
                    
                if self.ISUSD==True:
                    #print('USD mult ',self.usd_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                    self.cfs[j]*=self.usd_values.loc[self.observ_dates[j]]
                    #print('cfs after USD',self.cfs)
                
                elif self.ISEUR==True:
                    #print('EUR mult ',self.eur_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                    self.cfs[j]*=self.eur_values.loc[self.observ_dates[j]]
                    #print('cfs after EUR',self.cfs)
        
        return self.cfs    
        
        
class bond_3(bond):
    spec_attr=['ISUSD','ISEUR','participation_K','interval']
    
    def __init__(self):
        super().__init__()
        self.TrueCount = 0
        self.cur_period=0
        self.obs_in_period=0
        self.init_period=0
        self.init_obs_in_period=0
        self.init_TrueCount = 0
        self.usd=1
        self.eur=1
        self.bond_type=3
        self.participation_K=0
        self.price_calc_type=''
        self.interval_type=[]
        self.plus_number = []
        self.minus_number = []

    def hist(self, rets, first_date, last_date):
        self.TrueCount = 0
        self.cur_period=0
        self.obs_in_period=0
        self.init_period=0
        self.init_obs_in_period=0
        self.init_TrueCount = 0
        self.usd=1
        self.eur=1
        self.start_value = self.prices.loc[self.start_observation]
        print('start value: ', self.start_value)
        self.dates_temp=[self.start_obs]
        if self.every_day_obs==False:
            self.dates_temp.extend(self.observ_dates)
            self.end_of_period=[0 for i in range(0,len(self.dates_temp))]
            k=0
            for j in range(1,len(self.dates_temp)):
                if (self.dates_temp[j]>self.coup_dates[k]) and \
                      (self.dates_temp[j-1]<=self.coup_dates[k]):
                    self.end_of_period[j-1]=1
                    k+=1
        else:
            self.dates_temp.extend(self.F_calendar[self.start_obs:self.observ_dates[-1]].index[1:].date)
            self.end_of_period=[0 for i in range(0,len(self.dates_temp))]
            k=0
            for j in range(1,len(self.dates_temp)):
                if (self.dates_temp[j]>self.observ_dates[k]) and \
                      (self.dates_temp[j-1]<=self.observ_dates[k]):
                    self.end_of_period[j-1]=1
                    k+=1
        self.end_of_period[-1]=1
        print('Концы периода',self.end_of_period)  
        self.cfs=[0 for j in range(len(self.coupon_dates))]
        
        self.cur_value = self.start_value
        print('self.cur_value', self.cur_value)
        self.massiv= self.cur_value*np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        print('self.massiv \n', self.massiv)
        print('self.prices \n',self.prices.loc[first_date:last_date])
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])
        self.calc(rets,first_date, last_date)
        #берем все стартовые цены по активам
        self.cur_value = self.massiv.loc[last_date,self.tickers]
        print('self.cur_value after scrap \n', self.cur_value)
        if self.ISUSD==True: self.usd=self.usd_values.loc[last_date]
        if self.ISEUR==True: self.eur=self.eur_values.loc[last_date]
        print('Price on analysis date (var: cur_value)',self.cur_value)
        print('Current period\'s index is (var: cur_period)' ,self.cur_period)
        print('Observations in this period passed (var: obs_in_period): ', self.obs_in_period, \
              ', of which ', self.TrueCount, 'were True (var: TrueCount)')
        if self.ISUSD==True: print('usd',self.usd)
        if self.ISEUR==True: print('eur',self.eur)
        self.init_period=self.cur_period
        self.init_obs_in_period=self.obs_in_period
        self.init_TrueCount=self.TrueCount
        
    def calc(self, rets, first_date, last_date):
        self.massiv= self.cur_value*np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        #for i in observations:
        #    print(self.massiv.loc[i:i])
        #print('self.massiv calc \n', self.massiv)
        self.cur_period=self.init_period
        self.TrueCount=self.init_TrueCount
        self.obs_in_period=self.init_obs_in_period
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])
        for j in range(1,len(self.dates_temp)):
            #print('date',self.dates_temp[j])
            #print('Observations in this period passed (var: obs_in_period): ', self.obs_in_period)
            if (first_date<self.dates_temp[j]) and (self.dates_temp[j]<=last_date):
                #print('price on this date ',self.massiv.loc[self.dates_temp[j]])
                self.obs_in_period+=1
                #положительные значения интервалов
                self.plus_number = [i for i in self.interval_type if i > 0]
                #отрицательные значения интервалов
                self.minus_number = [i for i in self.interval_type if i < 0]
                #счётчик тикеров
                self.count_tickers = 0
                for i in range(0, len(self.tickers)):
                    #print('Prices: ', self.massiv.loc[self.dates_temp[j], self.tickers])
                    #print("Current underlying price: ", self.massiv.loc[self.dates_temp[j], self.tickers[i]], \
                    #      'Low: ',self.minus_number[i]+self.start_value[i], 'High: ', self.plus_number[i]+self.start_value[i])
                    if (self.plus_number[i] + self.start_value[i] >= self.massiv.loc[self.dates_temp[j], self.tickers].values[i] >= self.minus_number[i] + self.start_value[i]) == True:
                        self.count_tickers += 1

                if self.count_tickers == len(self.tickers):
                    self.TrueCount+=1
                    #print('True')
                #if sum((price[i]>intervals[i][0]) and ())==len(tickers)
                
                if self.end_of_period[j]==1:
                    #print('obs in per ', self.obs_in_period, 'true ', self.TrueCount)
                    self.cfs[self.cur_period]=self.participation_K*self.TrueCount/self.obs_in_period*self.principal
                    #print('cfs before USD',self.cfs)
                    if self.ISUSD==True:
                        #print('USD mult ',self.usd_values.loc[self.dates_temp[j]])
                        # Учитываем долларовый риск по купону
                        #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                        self.cfs[self.cur_period]*=self.usd_values.loc[self.dates_temp[j]]
                        #print('cfs after USD',self.cfs)
                    elif self.ISEUR==True:
                        #print('EUR mult ',self.eur_values.loc[self.observ_dates[j]])
                        # Учитываем долларовый риск по купону
                        #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                        self.cfs[self.cur_period]*=self.eur_values.loc[self.dates_temp[j]]
                        #print('cfs after EUR',self.cfs)
                    #print(self.cfs)
                    self.cur_period+=1
                    #print('cur period ', self.cur_period)
                    self.obs_in_period=0
                    self.TrueCount=0                
        return self.cfs
        
        
        
class bond_4(bond):
    spec_attr=['participation_K','vol_days','target_vol','MaxExp','CDS_int','CDS','libor','W']
    def __init__(self):
        super().__init__()
        self.bond_type=4
        self.participation_K=0
        self.CDS_int=[]
        self.target_vol=0
        self.W=pd.DataFrame([0])
        self.MaxExp=0
    def hist(self, rets, first_date, last_date):
        self.df=pd.concat([pd.DataFrame(columns=[i for i in self.tickers]), \
                           pd.DataFrame(columns=['CDS']), \
                           pd.DataFrame(columns=['deltaD' + i for i in self.W.columns]), \
                           pd.DataFrame(columns=['DK','volatility','exp','Lib1D','deltaZDK','ZDK'])],sort=False)
        self.cfs=[0 for j in range(len(self.coupon_dates))]
        #self.init_start_libor = self.libor
        self.init_start_cds=self.prices.shift().loc[self.st_date,'CDS']
        self.calc(pd.DataFrame(rets, copy=True),first_date, last_date)
        #self.init_start_libor = self.prices.shift().loc[self.ansys_date,'LIBOR']
        self.init_start_cds=self.prices.shift().loc[self.ansys_date,'CDS']
        self.df=self.anl_data

        
    def calc(self, rets, first_date, last_date):          
        #self.start_libor = self.init_start_libor
        self.start_cds=self.init_start_cds
        #rets.loc[:,'LIBOR'] = self.start_libor*np.exp(rets.loc[rets.index[0]:,'LIBOR'].cumsum()) # меняем #прогнозируемые доходности на значения
        rets.loc[:,'CDS'] = self.start_cds*np.exp(rets.loc[rets.index[0]:,'CDS'].cumsum()) # меняем прогнозируемые доходности на значения
        rets.loc[:,self.tickers] = np.exp(rets.loc[:,self.tickers].cumsum())
        rets.loc[:,self.tickers]=rets.loc[:,self.tickers]/rets.loc[:,self.tickers].shift()-1
        self.anl_data=pd.concat([self.df,rets.iloc[1:]],sort=False)
        for i in self.W.columns:
            self.anl_data['deltaD'+i]=self.anl_data[self.tickers].values.dot(self.W[i].values)
        for i in range(len(self.W.columns)):
            self.anl_data.loc[(self.anl_data['CDS']>self.CDS_int[i]) & (self.anl_data['CDS']<=self.CDS_int[i+1]), 'DK'] = \
                self.anl_data['deltaD'+self.W.columns[i]]

        if len(self.vol_days)==1:
            for i in range(len(self.W.columns)):
                self.anl_data.loc[(self.anl_data['CDS']>self.CDS_int[i]) & (self.anl_data['CDS']<=self.CDS_int[i+1]), 'volatility'] = \
                    self.vol(self.anl_data['deltaD'+self.W.columns[i]],self.vol_days[0])
           #   print('self.vol',self.vol(self.anl_data['deltaD'+self.W.columns[i]],self.vol_days[0]))
        else:
            for i in range(len(self.W.columns)):        
                self.anl_data.loc[(self.anl_data['CDS']>self.CDS_int[i]) & (self.anl_data['CDS']<=self.CDS_int[i+1]), 'volatility'] = \
                    np.maximum(*[self.vol(self.anl_data['deltaD'+self.W.columns[i]],N) for N in self.vol_days])
           
        #print('vol',self.anl_data['volatility'])
        self.anl_data['exp']=np.minimum(self.MaxExp,(self.target_vol)/self.anl_data['volatility'].shift())
        self.anl_data['Lib1D']=self.anl_data.index.date-np.roll(self.anl_data.index.date,1)
        self.anl_data['Lib1D']=self.anl_data['Lib1D'].dt.days/360*self.libor
        
        self.anl_data['deltaZDK']=1+self.anl_data['exp'].shift()*self.anl_data['DK']-self.anl_data['exp'].shift()*self.anl_data['Lib1D']
        
        self.anl_data['ZDK']=self.anl_data['deltaZDK'].cumprod()
        
        self.anl_data['ZDK']/=self.anl_data.loc[self.start_obs,'ZDK']
        #print('self.anl_data',self.anl_data)
        
        
        if self.observ_dates[-1] in self.anl_data.index:
            #print('np.maximum(0,self.anl_data.loc[self.observ_dat',self.anl_data)
            self.cfs = [self.principal*self.participation_K*np.maximum(0,self.anl_data.loc[self.observ_dates[-1],'ZDK']-1)]
        
        return self.cfs
    
    def vol(self,rets,N):
        return np.log(rets+1).rolling(N).std()*(252**0.5)
        

class bond_5(bond):        
    spec_attr=['ISUSD','ISEUR','coupon_rate','thresholds', ...]
    
    def __init__(self):
        super().__init__()
        self.init_not_paid=0
        self.usd=1
        self.eur=1
        self.bond_type=5
        #self.end_payment == False
    
    def hist(self, rets, first_date, last_date):
        self.init_not_paid=0
        self.usd=1
        self.eur=1
        self.cfs=[0 for j in range(len(self.coupon_dates))]
        self.hist_prices = [1 for i in range(len(self.tickers))]
        #print('self.hist_prices\n', self.hist_prices)
        #print('rets', np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers]))
        self.massiv = self.hist_prices * np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        print('Calc prices\n', self.massiv.loc[last_date,self.tickers]*self.prices.loc[self.start_obs,self.tickers].values)
        print('Fact prices\n', self.prices.loc[last_date,self.tickers])
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])    
        self.calc(rets,first_date, last_date)
        self.hist_prices=self.massiv.loc[last_date,self.tickers]
        if self.ISUSD==True: self.usd=self.usd_values.loc[last_date]
        if self.ISEUR==True: self.eur=self.eur_values.loc[last_date]
        self.init_not_paid=self.not_paid
        print('init_not_paid', self.init_not_paid)
        print('hist_prices:\n', self.hist_prices)
        if self.ISUSD==True: print('usd',self.usd)
        if self.ISEUR==True: print('eur',self.eur)
        
    def calc(self, rets, first_date, last_date):
        self.massiv= self.hist_prices*np.exp(rets.loc[first_date:last_date, self.tickers].cumsum(axis=0)-rets.loc[first_date, self.tickers])
        #print(self.massiv.loc[:,self.tickers]*self.prices.loc[self.start_obs,self.tickers].values)
        if self.ISUSD==True:
            self.usd_values=self.usd*np.exp(rets.loc[first_date:last_date, 'USDRUB'].cumsum(axis=0)-rets.loc[first_date, 'USDRUB'])
        if self.ISEUR==True:
            self.eur_values=self.eur*np.exp(rets.loc[first_date:last_date, 'EURRUB'].cumsum(axis=0)-rets.loc[first_date, 'EURRUB'])
        self.not_paid=self.init_not_paid
        
        for j in range(len(self.observ_dates)):
            if (first_date<self.observ_dates[j]) and (self.observ_dates[j]<=last_date): 
                self.cfs[j] = 0
        for j in range(len(self.observ_dates)):
            #print('date',self.observ_dates[j])

            if (first_date<self.observ_dates[j]) and (self.observ_dates[j]<=last_date):            
                #print('Цены активов\n', self.massiv.loc[self.observ_dates[j], self.tickers])
                
                #автоколл
                if self.autocall == True and sum(self.massiv.loc[self.observ_dates[j], self.tickers] > self.par[j][0][0]) >= self.par[j][2][0]:
                    #print('autocall triggered', ' barrier',self.par[j][0][0], 'n stocks',self.par[j][2][0])
                    self.cfs[j] = (self.par[j][1][0]+1) * self.principal + self.not_paid
                    #print('cfs bef USD',self.cfs)
                    if self.ISUSD==True:
                        #print('USD mult ',self.usd_values.loc[self.observ_dates[j]])
                        # Учитываем долларовый риск по купону
                        self.cfs[j]*=self.usd_values.loc[self.observ_dates[j]]
                        #print('cfs after USD',self.cfs, '\n')
                    elif self.ISEUR==True:
                        #print('EUR mult ',self.eur_values.loc[self.observ_dates[j]])
                        # Учитываем долларовый риск по купону
                        #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                        self.cfs[j]*=self.eur_values.loc[self.observ_dates[j]]
                        #print('cfs after EUR',self.cfs)
                    self.not_paid=0
                    #print('self.cfs', self.cfs)
                    return self.cfs
                
                #расчет потоков при условии пробития купонного барьера
                for i in range(len(self.par[j][0])):
                    #print('coup barrier',self.par[j][0][i],' n stocks',self.par[j][2][i])
                    if sum(self.massiv.loc[self.observ_dates[j], self.tickers] > self.par[j][0][i]) >= self.par[j][2][i]:
                        self.cfs[j] = self.par[j][1][i] * self.principal + self.not_paid
                        self.not_paid = 0
                        break
                    
                if self.cfs[j] == 0 and self.mem == True:
                    self.not_paid += self.par[j][1][-1] * self.principal
                    #print('add self.not_paid',self.not_paid)
                
                #расчет поставки худшего актива в дату погашения
                if self.observ_dates[j] == self.observ_dates[-1]:
                    #print('last date','delivery barrier',self.worst_delivery[0])
                    #print('self.worst_delivery[1]',self.worst_delivery[1],'\nstocks returns\n',self.massiv.loc[self.observ_dates[j], self.tickers]) 
                    if self.worst_delivery[1] > 0 and sum(self.massiv.loc[self.observ_dates[j], self.tickers] < self.worst_delivery[0]) >= self.worst_delivery[1]:
                        #определение n худшего актива и его поставка
                        #print("Сортировка \n", self.massiv.loc[self.observ_dates[j], self.tickers].sort_values())
                        #print('[self.worst_number-1]\n', [self.worst_number-1])
                        # * (1 / self.worst_delivery[0]) - добавлено для расчета сбер киб, "частичная защита капитала" при падении худшего актива
                        self.price_sort = self.massiv.loc[self.observ_dates[j], self.tickers].sort_values()[self.worst_number - 1] * (1 / self.worst_delivery[0])
                        #print('№ worst \n', self.price_sort)
                        self.cfs[-1] += np.maximum(self.protection, self.price_sort)*self.principal
                        #print('maximum\n',np.maximum(self.protection, (self.price_sort-1)))
                    else:
                        self.cfs[-1] += self.principal#*self.protection
                        

                #print('cfs bef USD',self.cfs[j])   
                if self.ISUSD==True:
                    #print('USD mult ',self.usd_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    self.cfs[j]*=self.usd_values.loc[self.observ_dates[j]]
                    #print('cfs after USD',self.cfs[j], '\n')
                elif self.ISEUR==True:
                    #print('EUR mult ',self.eur_values.loc[self.observ_dates[j]])
                    # Учитываем долларовый риск по купону
                    #self.cfs[j]*=(1.0276)**(dates_years[j+1]) #старый расчет
                    self.cfs[j]*=self.eur_values.loc[self.observ_dates[j]]
                    #print('cfs after EUR',self.cfs[-1])
            #print('date',self.observ_dates[j],'self.cfs',self.cfs)
            #если дополнительный доход выплачивается в конце срока обращения
            if self.observ_dates[j] == self.observ_dates[-1] and self.end_payment == True:
                a = sum(self.cfs)
                self.cfs = [0 for j in range(len(self.coupon_dates))]
                self.cfs[-1] = a
        
        return self.cfs
    
class MC:
    def __init__(self):
        self.n_windows = 0
        self.n_scenarios = 0
        self.mu = 0
    
    def calcMU(self,obj):
        mu = dict()
        usual_rets = obj.usual_rets.iloc[-250:-1]
        print("Calculating Mu")
        sp.cr_rates(obj)
        print('Covariance matrix for quanto adjustment\n',usual_rets.cov())
        for ticker in obj.tic_curr.keys():
            print(ticker)
            if obj.tic_curr[ticker][0] == 'USD':
                mu[ticker] = obj.USD_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'EUR':
                mu[ticker] = obj.EUR_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'RUB':
                mu[ticker] = obj.RUB_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'JPY':
                mu[ticker] = obj.JPY_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'HKD':
                mu[ticker] = obj.HKD_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'GBP':
                mu[ticker] = obj.GBP_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'CHF':
                mu[ticker] = obj.CHF_rate
                print('ri=',mu[ticker])
            if obj.tic_curr[ticker][0] == 'NOK':
                mu[ticker] = obj.NOK_rate
                print('ri=',mu[ticker])
            
            if  obj.tic_curr[ticker][0] != obj.Base_Curr:
                mu[ticker] = mu[ticker] - usual_rets[[ticker,obj.tic_curr[ticker][0]+obj.Base_Curr]].cov().iloc[1,0]
                print('quanto', - usual_rets[[ticker,obj.tic_curr[ticker][0]+obj.Base_Curr]].cov().iloc[1,0])
            
            if obj.tic_curr[ticker][1] == '1' or obj.tic_curr[ticker][1] == '2':
                #mu[ticker] = mu[ticker] - obj.RUB_rate
                if ticker[:3] == 'USD':
                    mu[ticker] = mu[ticker] - obj.USD_rate
                    print('rcurr =', obj.USD_rate)
                if ticker[:3] == 'EUR':
                    mu[ticker] = mu[ticker] - obj.EUR_rate
                    print('rcurr =', obj.EUR_rate)
                if ticker[:3] == 'RUB':
                    mu[ticker] = mu[ticker] - obj.RUB_rate
                    print('rcurr =', obj.RUB_rate)
                if ticker[:3] == 'JPY':
                    mu[ticker] = mu[ticker] - obj.JPY_rate
                    print('rcurr =', obj.JPY_rate)
                if ticker[:3] == 'HKD':
                    mu[ticker] = mu[ticker] - obj.HKD_rate
                    print('rcurr =', obj.HKD_rate)
                if ticker[:3] == 'GBP':
                    mu[ticker] = mu[ticker] - obj.GBP_rate
                    print('rcurr =', obj.GBP_rate)
                if ticker[:3] == 'CHF':
                    mu[ticker] = mu[ticker] - obj.CHF_rate
                    print('rcurr =', obj.CHF_rate)
                if ticker[:3] == 'NOK':
                    mu[ticker] = mu[ticker] - obj.NOK_rate
                    print('rcurr =', obj.NOK_rate)
            mu[ticker] = mu[ticker] - 0.5* usual_rets[ticker].std()**2
            print('-0.5*sigma**2 = ',- 0.5* usual_rets[ticker].std()**2)
        print(mu)
        return mu
            
    def run_MC(self,obj):
        print('\nMonte-Carlo is launched\n')
        num_samples = obj.days_to_generate
        obj.f_prices = []
        obj.cfs_ytm = []
        counter = np.linspace(0,100,11)
        obj.cols = obj.tic_curr.keys()
        c=0
        perc_count = obj.n_scenarios*obj.n_windows/100
        #Число прогнозных точек
        #if we need to calc FV
        if self.FV == True:
            print("Running scenario analysis for Fair Value calculation\n")
            # какую порцию доходностей используем
            usual_rets = obj.usual_rets.iloc[-250:-1]
            mu = self.calcMU(obj)
            mu = pd.Series(mu)
            # какая базовая ставка 
            #if значение виджета по валюте =...
           # if curr_wid.value == 'Валюта':
            sigma = usual_rets.std()
            obj.base_rate = obj.rates[obj.Base_Curr][0]
        # calc sigmas
        #строим мю:
        #mu=ri-rcurr-quanto-0.5*sigma**2   (rho*sigma*sigma_curr)
        #1 слагаемое - безрисковая ставка валюты актива
        #2 слагаемое - ставка иностр валюты
        # 3 слагаемое - кванто поправка
        # 4 слагаемое - поправка для расчета лог доходностей
            #print(usual_rets)
            rv_generator = MultiJSU(usual_rets, mean=self.mu, sigma=sigma, normal=True)                # Инициализируем генератор (подбираем распределения)
            #print('mu.keys()',mu.keys())
            rv = pd.DataFrame(rv_generator.generate(num_samples*obj.n_windows*obj.n_scenarios).T, columns=mu.keys()) # Генерируем массив доходностей
            #print(len(rv))
            #print('rv_generator', rv)
            obj.rv=rv
            #Changed
            for i in range(obj.n_windows*obj.n_scenarios):                                   # Цикл для расчета купонов
                if (i + 1) >= counter[c]*perc_count:
                    print(str(int(counter[c]))+ '% completed')
                    c+=1
                rv_samp = rv.iloc[num_samples*i:num_samples*(i+1)].reset_index() # Берем порцию доходностей, соответствующую сценарию i
                #print('rv_samp 859', rv_samp)
                rv_samp = rv_samp.merge(obj.F_calendar[obj.ansys_date:obj.MaxOf2].reset_index()['Date'], left_index=True, right_index=True) # +даты
                #print('rv_samp 861', rv_samp)
                rv_samp = rv_samp.drop(rv_samp.columns[[0]], axis='columns').set_index('Date') # очищаем массив, индекс = даты
                #print('rv_samp 863', rv_samp)
                rv_samp.columns=obj.cols
                obj.cf_sim = obj.calc(rv_samp,obj.ansys_date, obj.observ_dates[-1])                                    # Расчет купонов
                #print('obj.cf_sim 866', obj.cf_sim)
                obj.f_prices.extend(obj.cf_sim)                                    # Выделяем 1 поток купонов для расчета fair_price
                #print('obj.f_prices 868', obj.f_prices)
            obj.f_prices_matrix = [obj.f_prices[i:i + len(obj.cf_sim)] for i in range(0, len(obj.f_prices), len(obj.cf_sim))]
            #print('obj.f_prices_matrix 870', obj.f_prices_matrix)
            obj.opt_payoff = pd.DataFrame(obj.f_prices_matrix).mean() #  усредненные по всем сценариям купоны 
            #print('obj.opt_payoff 872', obj.opt_payoff)
            print("Scenario analysis for Fair Value calculation is completed\n")

        if self.Ret == True:
        #elif we need to calc expected returns
            print("Running scenario analysis for Expected Return calculation") 
            c = 0
            for k in range(obj.n_windows):

                #print('Calculating window ',k+1,' out of ',obj.n_windows)                # Счетчик расчета окон
                used_rets = obj.rets.iloc[obj.windows_start[k]:obj.windows_start[k]+obj.window]        # выбираем нужное окно исторических доходностей
                try:
                    rv_generator = MultiJSU(used_rets, normal=obj.normal)                # Инициализируем генератор (подбираем распределения)
                except Exception:
                    rv_generator = MultiJSU(used_rets, normal=True) 
                
                rv = pd.DataFrame(rv_generator.generate(num_samples*obj.n_scenarios).T, columns=obj.rets.columns) # Генерируем массив доходностей
                obj.rv=rv
                for i in range(obj.n_scenarios):                                   # Цикл для расчета купонов
                    if (k+1)*(i+1)>=counter[c]*perc_count:
                        print(str(int(counter[c]))+ '% completed')
                        c+=1
                    rv_samp = rv.iloc[num_samples*i:num_samples*(i+1)].reset_index() # Берем порцию доходностей, соответствующую сценарию i
                    rv_samp = rv_samp.merge(obj.F_calendar[obj.ansys_date:obj.MaxOf2].reset_index()['Date'], left_index=True, right_index=True) # +даты
                    rv_samp = rv_samp.drop(rv_samp.columns[[0]], axis='columns').set_index('Date') # очищаем массив, индекс = даты
                    rv_samp.columns=obj.cols

                    obj.cf_sim = obj.calc(rv_samp,obj.ansys_date,obj.observ_dates[-1])                                    # Расчет купонов

                    obj.cfs_ytm.extend(obj.cf_sim)                                     # Выделяем 2 поток купонов для расчета YTM
                    #print(obj.cf_sim)
            # преобразуем в матрицу Число сценариев Х Число потоков
            print("Scenario analysis for Expected Return calculation is completed")                
        
'''
Классы, отвечающие за моделирование распределения JohnsonSU
Адаптирован код, размещенный здесь: https://github.com/chrsbats/connorav/blob/master/README.md
'''
NORMAL_CUTOFF = 0.01
class MSSKDistribution(object):
  
    def __init__(self, mean=None, std=None, skew=None, kurt=None):
        if isinstance(mean,np.ndarray) or mean != None:
            self.fit(mean,std,skew,kurt)

    def fit(self, mean, std=None, skew=None, kurt=None):
        if std == None:
            #Array or tuple format.
            self.m = mean[0]
            self.s = mean[1]
            self.skew = mean[2]
            self.kurt = mean[3]
        else:
            self.m = mean
            self.s = std
            self.skew = skew
            self.kurt = kurt

        if abs(self.skew) < NORMAL_CUTOFF and abs(self.kurt) < NORMAL_CUTOFF:  
            #It is hard to solve the johnson su curve when it is very close to normality, so just use a normal curve instead.
            self.dist = norm(loc=self.m,scale=self.s)
            self.skew = 0.0
            self.kurt = 0.0

        else:
            a,b,loc,scale = self._johnsonsu_param(self.m,self.s,self.skew,self.kurt)
            self.dist = johnsonsu(a,b,loc=loc,scale=scale)

    def _optimize_w(self,w1,w2,b1,b2):
        def m_w(w):
            m = -2.0 + np.sqrt( 4.0 + 2.0 * ( w ** 2.0 - (b2 + 3.0) / (w ** 2.0 + 2.0 * w + 3.0)))
            return m

        def f_w(w):
            m = m_w(w)
            fw = (w - 1.0 - m) * ( w + 2.0 + 0.5 * m) ** 2.0
            return (fw - b1) ** 2.0

        
        if abs(w1 - w2) > 0.1e-6:
            solution = minimize_scalar(f_w, method='bounded',bounds=(w1,w2))
            w = solution['x']
        else:
            if w1 < 1.0001:
                w = 1.0001
            else:
                w = w1

        m = m_w(w)    

        return w, m


    def _johnsonsu_param(self,mean,std_dev,skew,kurt):
        #"An algorithm to determine the parameters of SU-curves in the johnson system of probabillity distributions by moment matching", HJH Tuenter, 2001
        
        #First convert the parameters into the moments used by Tuenter's alg. 
        u2 = std_dev ** 2.0
        u3 = skew * std_dev ** 3.0
        u4 = (kurt + 3.0) * std_dev ** 4.0
        b1 = u3 ** 2.0 / u2 ** 3.0
        b2 = kurt + 3.0

        w2 = np.sqrt((-1.0 + np.sqrt(2.0 * (b2 -1.0))))
        big_d = (3.0 + b2) * (16.0 * b2 * b2 + 87.0 * b2 + 171.0) / 27
        d = -1.0 + (7.0 + 2.0 * b2 + 2.0 * np.sqrt(big_d)) ** (1.0 / 3.0) - (2.0 * np.sqrt(big_d) - 7.0 - 2.0 * b2) ** (1.0 / 3.0)
        w1 = (-1.0 + np.sqrt(d) + np.sqrt( 4 / np.sqrt(d) - d - 3.0)) / 2.0
        if (w1 - 1.0) * ((w1 + 2.0) ** 2.0) < b1:
            #no curve will fit
            raise Exception("Invalid parameters, no curve will fit")

        w, mw = self._optimize_w(w1,w2,b1,b2)

        z = ((w + 1.0) / (2.0 * w )) * ( ((w - 1.0) / mw) - 1.0) 
        if z < 0.0:
            z = 0.0
        omega = -1.0 * np.sign(u3) * np.arcsinh(np.sqrt(z))
        
        a = omega / np.sqrt(np.log(w))
        b = 1.0 / np.sqrt(np.log(w))

        z =  w - 1.0 - mw
        if z < 0.0:
            z = 0.0
        loc = mean - np.sign(u3) * (std_dev / (w -1.0)) * np.sqrt(z)
        
        scale = std_dev / (w - 1.0) * np.sqrt( (2.0 * mw) / ( w + 1.0))

        return a,b,loc,scale

    def _cvar(self,upper=0.05,samples=64,lower=0.00001):
        interval = (upper - lower) / float(samples)
        ppfs = self.dist.ppf(np.arange(lower, upper+interval, interval))
        result = integrate.romb(ppfs, dx=interval)
        return result
    
    #Visible scipy methods for distribution objects. 
    #Note that scipy uses some funky metaprogramming.  It's easier to do this than to inherit from rv_continuous.
    def rvs(self,x=None):
        return self.dist.rvs(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def logcdf(self, x):
        return self.dist.logcdf(x)

    def sf(self, x):
        return self.dist.sf(x)

    def logsf(self, x):
        return self.dist.logsf(x)

    def ppf(self, x):
        return self.dist.ppf(x)

    def isf(self, x):
        return self.dist.isf(x)

    def mean(self):
        return self.dist.mean()

    def median(self):
        return self.dist.median()
    
    def std(self):
        return self.dist.std()

    def var(self):
        return self.dist.var()

    def stats(self):
        return self.m, self.s, self.skew, self.kurt
    


class MultiJSU(object):
    import pandas as pd
    def __init__(self,data,mean=None,sigma=pd.Series(-1),method='cholesky',normal=False):
        
        if mean is None:
            if normal == False:
                stats_init=pd.DataFrame({'Mean':data.mean(),
                          'Std':data.std(),
                          'Skewness':skew(data),
                          'Kurtosis':kurtosis(data)})
            else:
                stats_init=pd.DataFrame({'Mean':data.mean(),
                          'Std':data.std(),
                          'Skewness':0,
                          'Kurtosis':0})
        elif normal == True:
            stats_init=pd.DataFrame({'Mean':mean,
                          'Std':sigma,
                          'Skewness':0,
                          'Kurtosis':0})
        else:
            stats_init=pd.DataFrame({'Mean':mean,
                          'Std':sigma,
                          'Skewness':skewness(data),
                          'Kurtosis':kurtosis(data)})

#         if normal==True:
#             stats_init['Skewness']=0
#             stats_init['Kurtosis']=0
        self.moments=np.asarray(stats_init)
        self.correlations=data.corr()      
        self.dimensions = self.correlations.shape[0]##Изменено
        self.distributions = [MSSKDistribution(self.moments[i]) for i in range(data.shape[1])]#Изменено

        
        
    def JSU_fit(self,data):
        a,b,loc,scale=johnsonsu.fit(data)
        return johnsonsu(a,b,loc,scale)    

    def generate(self,num_samples,method='cholesky'):
        rv = self._uniform_correlated(self.dimensions,self.correlations,num_samples,method)
        rv = rv.tolist()
        for d in range(self.dimensions):
            rv[d] = self.distributions[d].ppf(rv[d])
        self.rv = np.array(rv)
        return self.rv
        

    def _normal_correlated(self,dimensions,correlations,num_samples,method='cholesky'):

        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        x = norm.rvs(size=(dimensions, num_samples))

        # We need a matrix `c` for  which `c*c^T = r`.  We can use, for example,
        # the Cholesky decomposition, or the we can construct `c` from the
        # eigenvectors and eigenvalues.
        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(correlations, lower=True)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correlations)
            # Construct c, so c*c^T = r.
            c = np.dot(evecs, np.diag(np.sqrt(evals)))

        # Convert the data to correlated random variables. 
        y = np.dot(c, x)
        return y

    def _uniform_correlated(self,dimensions,correlations,num_samples,method='cholesky'):
        #print(correlations)
        #корректируем корреляционную матрицу, чтобы после генерации случайных чисел из 
        #нормального распределения корреляции были ближе к эмпирическим
        adj_corr = 2*np.sin((np.pi/6)*correlations)
        #print(adj_corr)
        normal_samples = self._normal_correlated(dimensions,adj_corr,num_samples,method) 
        x = norm.cdf(normal_samples)
        #print(np.corrcoef(x))
        return x












        