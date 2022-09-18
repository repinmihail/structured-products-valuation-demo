# Инициализация необх библиотек
from scipy.stats import johnsonsu, norm
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
import quandl
from urllib.request import urlopen
import json
import okama as ok


from bond import bond_1,bond_2, bond_3, bond_4, bond_5, MC
from widgets import *

global bond_type
global bond_counter
global bond
global model
model = MC()

bond_counter=1 #счетчик созданных облигаций
bond_list=[] #запоминание объектов созданных облигаций 
quandl.ApiConfig.api_key = 'vg_K--5SxeS86FWGTqMt'
pd.set_option('display.max_rows', 10)
plt.style.use('bmh')
today=str(date.today())
current_module = sys.modules[__name__]



def get_iss_price(ticker,board='TQBR', engine='stock', market='shares',start='2007-01-01', end=today):
    '''
    Загрузка исторических цен с Мосбиржи. По умолчанию грузит цены акций с 2007 года. Параметры по умолчанию:
    board='TQBR', 
    engine='stock', 
    market='shares',
    start='2007-01-01', 
    end=today
    '''
    with requests.Session() as session:
        data = apimoex.get_board_history(session, ticker, board=board,engine=engine, market=market, columns=['TRADEDATE','CLOSE'], start=start, end=end,)
        df = pd.DataFrame(data)
        df.rename(columns={'TRADEDATE': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        df.index=pd.to_datetime(df.index, format="%Y-%m-%d")
        df.columns=[ticker]
        return df

def cr_prices(obj):
  #  Выгрузка исторических цен базовых активов, курсов валют (если от них зависят выплаты) из разных источников:
 #   Yahoo (по умолчанию), Мосбиржи, файла CSV (в зависимости от значения атрибута obj.data_source: 'yahoo', 'moex' или название файла). #Результаты записываются в атрибут obj.prices. Загрузка происходит за максимальный период не ранее 01.01.2007.

    ###########################
    # Добавить загрузку нужных валют
    #########################
    print(obj.data_source)
    if obj.data_source=='yahoo':
        if obj.include_dividends==False:
            close_type='Close'
        else:
            close_type='Adj Close'
        #создаем массив цен для активов из словаря (цены из yahoo)
        obj.prices = pd.DataFrame(data = pdr.get_data_yahoo(list(obj.tic_curr)[0], start="2007-01-01", end=obj.analysis_date)[close_type])
        obj.prices.columns = [list(obj.tic_curr)[0]]
        if list(obj.tic_curr)[0] == 'USDRUB':
            obj.prices.loc['2016-01-06',['USDRUB']]=71.62
        if list(obj.tic_curr)[0] == 'RUBUSD':
            obj.prices.loc['2016-01-06',['RUBUSD']]=1/71.62
        if list(obj.tic_curr)[0] == 'GBPRUB':
            obj.prices.loc['2016-01-06',['GBPRUB']]=96.9970
        if list(obj.tic_curr)[0] == 'RUBGBP':
            obj.prices.loc['2016-01-06',['RUBGBP']]=1/96.9970
        #для каждого актива подгружаем валюты
        print('Assets from Yahoo: ', list(obj.tic_curr)[1:])
        for ticker in list(obj.tic_curr)[1:]:
            if ticker not in obj.asset_from_csv:
                if obj.tic_curr[ticker][1] == '0':
                    p=pd.DataFrame(data = pdr.get_data_yahoo(ticker, start="2007-01-01", end=obj.analysis_date)[close_type])
                    p.columns=[ticker]
                    obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)    
                elif obj.tic_curr[ticker][1] == '1' or obj.tic_curr[ticker][1] == '2':
                    if ticker == 'CHFRUB':
                        chf = ok.QueryData.get_close(symbol='CHFRUB.FX', first_date='2007-01', last_date=obj.analysis_date, period='D')
                        chf.to_excel('CHFRUB.xlsx', engine='xlsxwriter')
                        chfrub = pd.read_excel('CHFRUB.xlsx', sep=';')
                        chfrub['Date'] = pd.to_datetime(chfrub['Date'], format = '%d.%m.%Y')
                        chfrub.set_index('Date', inplace = True)
                        chfrub.columns = [ticker]
                        obj.prices=obj.prices.merge(chfrub,how='left',left_index=True,right_index=True)
                        #print(obj.prices)
                    else:
                        p = pd.DataFrame(pdr.get_data_yahoo(ticker+'=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
                        p.columns=[ticker]
                        obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)
                        if ticker == 'USDRUB':
                            obj.prices.loc['2016-01-06',['USDRUB']]=71.62
                        if ticker == 'RUBUSD':
                            obj.prices.loc['2016-01-06',['RUBUSD']]=1/71.62
                        if ticker == 'GBPRUB':
                            obj.prices.loc['2016-01-06',['GBPRUB']]=96.9970
                        if ticker == 'RUBGBP':
                            obj.prices.loc['2016-01-06',['RUBGBP']]=1/96.9970
                else:
                     p=pd.DataFrame(pdr.get_data_yahoo(ticker+'=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
                     p.columns=[ticker+obj.tic_curr[ticker][0]]
                     obj.prices=pd.concat([obj.prices,p],axis = 1)
                    
            # Загрузка дополнительных активов из csv (yahoo+csv). Актив(-ы), загружаемый(-ые) из файла csv должен(-ны) указываться НЕ первым(-и) в поле виджета
            else:
                assets_data = pd.read_excel('assets.xlsx', sheet_name=sp.bond.ISIN) #encoding = "ISO-8859-1"
                assets_data['Date'] = pd.to_datetime(assets_data['Date'], format = '%d.%m.%Y')
                assets_data.set_index('Date', inplace = True)
                assets_data = assets_data.dropna()
                print('Assets data from csv: \n',assets_data)
                for asset in obj.asset_from_csv:
                    obj.prices = obj.prices.merge(assets_data[asset], how='left', left_index=True, right_index=True)
                print('Prices after merge with CSV:\n',obj.prices)
                
        #print('prices:\n',obj.prices)
        obj.prices=obj.prices[obj.prices.dropna().iloc[1].name:]
        obj.prices=obj.prices.fillna(method='ffill')

        #На случай, если нужно загрузить курс доллара
        if obj.ISUSD==True and 'USDRUB' not in list(obj.prices):
            p=pd.DataFrame(pdr.get_data_yahoo('USDRUB=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
            p.columns=['USDRUB']
            obj.cols.extend(['USDRUB'])
            obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)
            obj.prices.loc['2016-01-06',['USDRUB']]=71.62
        if obj.ISEUR==True and 'EURRUB' not in list(obj.prices):
            p=pd.DataFrame(pdr.get_data_yahoo('EURRUB=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
            p.columns=['EURRUB']
            obj.cols.extend(['EURRUB'])
            obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)    
    else:
        obj.prices=pd.read_csv(obj.data_source,sep=';')
        obj.prices['Date'] = pd.to_datetime(obj.prices['Date'], format=obj.dateformat)
        obj.prices.set_index('Date', inplace=True)
        obj.prices.sort_index(inplace=True)
        obj.prices=obj.prices[:obj.analysis_date]  
        
        for ticker in list(obj.tic_curr)[1:]:
            if obj.tic_curr[ticker][1] == '2':
                if ticker == 'EURRUB':
                    eur = ok.QueryData.get_close(symbol='EURRUB.FX', first_date='2007-01', last_date=obj.analysis_date, period='D')
                    eur.to_excel('EURRUB.xlsx', engine='xlsxwriter')
                    eurrub = pd.read_excel('EURRUB.xlsx', sep=';')
                    eurrub['Date'] = pd.to_datetime(eurrub['Date'], format = '%d.%m.%Y')
                    eurrub.set_index('Date', inplace = True)
                    eurrub.columns = [ticker]
                    obj.prices=obj.prices.merge(eurrub,how='left',left_index=True,right_index=True)
                elif ticker == 'USDRUB':
                    usd = ok.QueryData.get_close(symbol='USDRUB.FX', first_date='2007-01', last_date=obj.analysis_date, period='D')
                    usd.to_excel('USDRUB.xlsx', engine='xlsxwriter')
                    usdrub = pd.read_excel('USDRUB.xlsx')
                    usdrub['Date'] = pd.to_datetime(usdrub['date'], format = '%d.%m.%Y')
                    usdrub.drop('date', axis=1, inplace=True)
                    usdrub.set_index('Date', inplace = True)
                    usdrub.columns = [ticker]
                    obj.prices=obj.prices.merge(usdrub,how='left',left_index=True,right_index=True)
                elif ticker == 'GBPRUB':
                    gbp = ok.QueryData.get_close(symbol='GBPRUB.FX', first_date='2007-01', last_date=obj.analysis_date, period='D')
                    gbp.to_excel('GBPRUB.xlsx', engine='xlsxwriter')
                    gbprub = pd.read_excel('GBPRUB.xlsx', sep=';')
                    gbprub['Date'] = pd.to_datetime(gbprub['Date'], format = '%d.%m.%Y')
                    gbprub.set_index('Date', inplace = True)
                    gbprub.columns = [ticker]
                    obj.prices=obj.prices.merge(gbprub,how='left',left_index=True,right_index=True)
                else:
                    p=pd.DataFrame(pdr.get_data_yahoo(ticker+'=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
                    p.columns=[ticker]
                    obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)
                    if ticker == 'USDRUB':
                        obj.prices.loc['2016-01-06',['USDRUB']]=71.62
                    if ticker == 'RUBUSD':
                        obj.prices.loc['2016-01-06',['RUBUSD']]=1/71.62
                    if ticker == 'GBPRUB':
                        obj.prices.loc['2016-01-06',['GBPRUB']]=96.9970
                    if ticker == 'RUBGBP':
                        obj.prices.loc['2016-01-06',['RUBGBP']]=1/96.9970
        #p=pd.DataFrame(pdr.get_data_yahoo('USDRUB=X', start="2007-01-01", end=obj.analysis_date)['Adj Close'])
        #p.columns=['USDRUB']
        #obj.cols.extend(['USDRUB'])
        #obj.prices=obj.prices.merge(p,how='left',left_index=True,right_index=True)
        #obj.prices.loc['2016-01-06',['USDRUB']]=71.62
    if len(obj.CDS)>0:
        obj.prices = obj.prices.merge(obj.CDS, how='right', left_index=True, right_index=True)
        obj.cols.extend(['CDS'])
        obj.prices = obj.prices[pd.isnull(obj.prices[obj.tickers[0]])==False]
        obj.tic_curr['CDS'] = ('USD','0')
    obj.prices=obj.prices[obj.prices.dropna().iloc[1].name:]
    obj.prices=obj.prices.fillna(method='ffill')
    

def check_prices(obj):
    '''
    Печатает нормированные цены активов, чтобы пользователь убедился в корректности их загрузки (отсутствие рывков, пропусков). Также печатает число получившихся торговых дней в одном из годов (2018) для проверки.
    '''
    #print('Проверьте корректность: \n' \
    #      'В 2018 получилось '+str(obj.prices['2018'].shape[0])+ ' торговых дней (должно быть 250-260)')
    scaler = MinMaxScaler()
    pd.DataFrame(data=scaler.fit_transform(obj.prices), index=obj.prices.index, columns=obj.prices.columns).plot(figsize=(20,7))
    plt.legend(facecolor='white', framealpha=1)
    plt.show()

def cr_rets(obj):
    '''
    Расчет исторических дневных логарифмических доходностей
    '''
    obj.rets=pd.DataFrame(
        np.log(obj.prices[1:].values / obj.prices[:-1].values), 
        columns=obj.prices.columns, 
        index=obj.prices.index[1:]
    ).dropna()
    #print('from cr_rets\n', obj.rets)
    #####################
    # Обычные доходности протестировать корректность
    #####################
    obj.usual_rets=pd.DataFrame(obj.prices[1:].values / obj.prices[:-1].values-1, 
                                columns=obj.prices.columns, 
                                index=obj.prices.index[1:]
    ).dropna()
    
def cr_calendar(obj):
    '''
    Из файла calendar.xls создает массив будущих рабочих дней. Зависит от атрибута объекта облигации base_source, в котором указывается страна базовых активов. Результат записывается в атрибут F_calendar.
    '''
    xl = pd.ExcelFile('calendar.xls')
    f_calendar = xl.parse('f_calendar')
    f_calendar = f_calendar.set_index('Date')
    obj.F_calendar = f_calendar[obj.base_source][(pd.isnull(f_calendar[obj.base_source])==True)&(f_calendar.index>=obj.analysis_date)] # только рабочие дни страны базовых активов
    obj.F_calendar=pd.Series(index=obj.prices.loc[:obj.analysis_date].index[:-1]).append(obj.F_calendar)

def cr_days_to_generate(obj):
    '''
    Рассчитывает ряд переменных, связанных с датами:
    obj.MaxOf2 (формат даты) - последняя дата, на которую может потребоваться спрогнозировать цены базовых активов. Максимум из последней даты наблюдения и последней даты выплаты купона.
    obj.ansys_date - дата анализа в формате даты, а не текста, как в атрибуте obj.analysis_date
    obj.start_obs- дата начального наблюдения в формате даты, а не текста, как в атрибуте obj.start_observation = 
    obj.observ_dates - список дат наблюдений в формате даты, а не текста, как в атрибуте obj.observations
    obj.coup_dates - список дат выплаты купонов в формате даты, а не текста, как в атрибуте obj.coupon_dates
    obj.days_to_generate - число дней, на которые моделируем цены (с даты анализа по дату в obj.MaxOf2)
    '''
    print("Cp = ",obj.coupon_dates[-1])
    print('obs =',obj.observations[-1])
    obj.MaxOf2 = max(dt.strptime(obj.coupon_dates[-1], '%Y-%m-%d').date(),dt.strptime(obj.observations[-1], '%Y-%m-%d').date())
    obj.ansys_date = dt.strptime(obj.analysis_date, '%Y-%m-%d').date() # дата анализа
    obj.start_obs = dt.strptime(obj.start_observation, '%Y-%m-%d').date()
    obj.observ_dates = [dt.strptime(i, '%Y-%m-%d').date() for i in obj.observations] #преобразуем формат из текста в даты
    obj.coup_dates = [dt.strptime(i, '%Y-%m-%d').date() for i in obj.coupon_dates]  #преобразуем формат из текста в даты
    obj.days_to_generate = obj.F_calendar[obj.ansys_date:obj.MaxOf2].shape[0]
    # расчет дней обращения структурного продукта для вычета комиссии
    obj.start_date_c = dt.strptime(obj.issue_date, '%Y-%m-%d').date() # дата выпуска
    obj.end_date_c = dt.strptime(obj.coupon_dates[-1], '%Y-%m-%d').date() # дата погашения
    obj.days_between = (obj.end_date_c - obj.start_date_c).days
    print('\nКоличество дней в обращении = ', obj.days_between)
    
    
def cr_dates_years(obj):
    '''
    Рассчитывает переменную obj.dates_years - список полных лет до каждой из дат выплат купонов. Используется в дисконтировании денежных потоков.
    '''
    y0,m0,d0=obj.analysis_date.split('-')
    da0 = date(int(y0), int(m0), int(d0))
    obj.dates_years=[]
    for i in range(0,len(obj.coupon_dates)):
        y,m,d=obj.coupon_dates[i].split('-')
        da0 = date(int(y0), int(m0), int(d0))
        da1 = date(int(y), int(m), int(d))
        obj.dates_years.append((da1-da0).days/365.25)
    obj.dates_years.insert(0,0)
    

def cr_rf_rate(obj):
    '''
    Создание списка российских безрисковых ставок с КБД ОФЗ (obj.rf_rate), соответствующих по сроку полным годам до выплаты купонов. Данные с Мосбиржи.
    '''
    obj.rf_rate = []
    for i in obj.dates_years[1:]: 
        if i < 0:
            obj.rf_rate.append(0)
        else:
            payload = {'date': obj.analysis_date, 'period': round(i,2)}  
            moex_answ = requests.get('https://iss.moex.com/iss/apps/bondization/zcyccalculator.json', params=payload)
            moex_data = moex_answ.json()['zcyc']
            obj.rf_rate.append(moex_data['data'][0][0]/100)

def cr_rates(obj):
    '''
    Выгрузка с Мосбиржи короткой ставки с КБД
    '''
    
    payload = {'date': obj.analysis_date, 'period': 0.001}  
    moex_answ = requests.get('https://iss.moex.com/iss/apps/bondization/zcyccalculator.json', params=payload)
    moex_data = moex_answ.json()['zcyc']
    obj.RUB_rate=moex_data['data'][0][0]/100
    #print('Рублёввая ставка', obj.RUB_rate)
    obj.USD_rate=quandl.get('FRED/TB3MS',start_date=dt(2007,1,1),end_date=obj.ansys_date).iloc[-1][0]/100
    obj.JPY_rate = -0.0002
    obj.EUR_rate=quandl.get(
        'BUNDESBANK/BBSIS_D_I_ZST_ZI_EUR_S1311_B_A604_R01XX_R_A_A__Z__Z_A',start_date=dt(2007,1,1),end_date=obj.ansys_date).iloc[-1][0]/100
    #download HIBOR for HKD
    if 'HKD' in list(obj.tic_curr.iloc[0]):
        url = 'https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hk-interbank-ir-daily?segment=hibor.fixing&pagesize=2000&offset=0&fields=end_of_day,ir_overnight&choose=end_of_day&from=2007-01-01&to=2021-01-12'
        with urlopen(url) as r:
            HKD = json.loads(r.read().decode(r.headers.get_content_charset('utf-8')))
        for i in HKD['result']['records']:
            if i['end_of_day'] == obj.analysis_date:
                obj.HKD_rate = i['ir_overnight']/100
    else:
        obj.HKD_rate = 0
    #download SONIA for GBP
    if 'GBP' in list(obj.tic_curr.iloc[0]):
        GBP_rate = pd.read_csv('GBP_Sonia.csv', sep=';')
        GBP_rate['Date'] = pd.to_datetime(GBP_rate['Date'], format = '%d.%m.%Y')
        GBP_rate.set_index('Date', inplace = True)
        obj.GBP_rate = GBP_rate.loc[obj.analysis_date, 'GBP_Sonia'].values[0]/100
    else:
        obj.GBP_rate = 0
    #download SARON for CHF
    if 'CHF' in list(obj.tic_curr.iloc[0]):
        CHF_rate = pd.read_csv('CHF_SARON.csv', sep=';')
        CHF_rate['Date'] = pd.to_datetime(CHF_rate['Date'], format = '%d.%m.%Y')
        CHF_rate.set_index('Date', inplace = True)
        if obj.analysis_date in CHF_rate.index:
            obj.CHF_rate = CHF_rate.loc[obj.analysis_date, 'CHF_SARON'].values[0]/100
        else:
            obj.CHF_rate = CHF_rate.iloc[-1].values[0]/100
    else:
        obj.CHF_rate = 0
    
    #download NOWA for NOK
    if 'NOK' in list(obj.tic_curr.iloc[0]):
        NOK_rate = pd.read_csv('NOK_Nowa.csv', sep=';')
        NOK_rate['Date'] = pd.to_datetime(NOK_rate['Date'], format = '%d.%m.%Y')
        NOK_rate.set_index('Date', inplace = True)
        if obj.analysis_date in NOK_rate.index:
            obj.NOK_rate = NOK_rate.loc[obj.analysis_date, 'NOK_Nowa'].values[0]/100
        else:
            obj.NOK_rate = NOK_rate.iloc[-1].values[0]/100
    else:
        obj.NOK_rate = 0
        
    obj.rates = pd.DataFrame(
        {'RUB':[obj.RUB_rate], 'USD':[obj.USD_rate], 'EUR':[obj.EUR_rate],
         'JPY':[obj.JPY_rate], 'HKD':[obj.HKD_rate], 'GBP':[obj.GBP_rate],
         'CHF':[obj.CHF_rate], 'NOK':[obj.NOK_rate]}
    )
    print('Ставки\n', obj.rates)
    #print('Рассчёт производится с зашумлёнными данными по ставкам')
#     obj.rates['RUB']+=0.005
#     obj.rates['USD']-=0.005
                     
def cr_USD_IRate(obj):
    '''
    Выгрузка с Мосбиржи индикативной ставки доходности по свопу USD_1Y (атрибут obj.USD_IRate)
    '''
    request_url = ('https://iss.moex.com/iss/statistics/engines/currency/markets/fixing/SRATE_USD_1Y.json')

    an_d = (dt.strptime(obj.analysis_date, '%Y-%m-%d') - timedelta(days=30)).date() # сдвиг начальной даты чтобы избежать выходных
    arguments = {'from': an_d, 'till': obj.analysis_date}

    with requests.Session() as session:
        iss = apimoex.ISSClient(session, request_url, arguments).get()
        US_frame = pd.DataFrame(iss['history'])
        US_frame.set_index('tradedate', inplace=True)
        obj.USD_IRate = US_frame['rate'].iloc[-1:].to_numpy().item()/100
    
def cr_windows(obj):
    '''
    Создание окон, на которые будет нарезана история.
    Длина каждого окна (obj.window) = max(250 дней;оставшийся срок до погашения облигации (obj.days_to_generate))
    Если число окон равно 1, длина окна берется равной всему историческому периоду.
    '''
    obj.window=max(250,int(obj.days_to_generate))#срок до погашения облигации
    obj.windows_start=np.linspace(0,obj.rets.shape[0]-obj.window,obj.n_windows,dtype=int) # выбор окон из массива реальных доходностей
    if obj.n_windows==1: obj.window=obj.rets.shape[0]

def cr_hist_yields(obj):
    '''
    Обрезание истории доходностей до периода с даты эмиссии по анализируемую дату -1 день (атрибут obj.yields). Атрибут используется в расчете исторических купонов.
    '''
    if len(obj.vol_days)==0:
        obj.st_date=obj.start_obs
    elif len(obj.vol_days)==1:
        obj.st_date=obj.rets.loc[:obj.start_obs].index[-obj.vol_days[0]-3].date()
    else:
        obj.st_date=obj.rets.loc[:obj.start_obs].index[-max(obj.vol_days)-3].date()
    obj.yields = obj.rets.loc[min(obj.st_date,dt.strptime(obj.issue_date, '%Y-%m-%d').date()).strftime('%Y-%m-%d'):(obj.ansys_date).strftime('%Y-%m-%d')] 
    #print(rets.iloc[windows_start[0]:windows_start[0]+window], '\n', yields) # 1 окно и используемые ральные дох-ти
    #print(yields.iloc[:1,].to_numpy().item())



def npv(irr, cfs, yrs):
    '''
    Расчет дисконтированной стоимости
    '''
    return np.sum(cfs / (1. + irr) ** yrs)

def irr(cfs, yrs, x0=0, **kwargs):
    '''
    Расчет внутренней нормы доходности (доходности к погашению)
    '''
    cfs_local=[]
    yrs_local=[]
    for i in range(len(cfs)):
        if yrs[i]<0:
            yrs_local.extend([0])
            cfs_local.extend([0])
        else:
            yrs_local.extend([yrs[i]])
            cfs_local.extend([cfs[i]])
    return fsolve(npv, x0=x0, args=(np.asarray(cfs_local), np.asarray(yrs_local)), **kwargs).item()                     

                     
def run_analysis(bond,print_output=True):
    '''
    Запуск цикла расчета справедливой стоимости и доходности к погашению. При запуске функций по умолчанию происходит печать основных переменных, создаваемых функциями, для проверки корректности их работы.
    Запускаются следующие функции:
    cr_calendar(bond) - создание календаря рабочих дней
    cr_days_to_generate(bond) - форматирование части дат в формат даты и расчет числа дней, на которые будем моделировать цены базовых активов
    cr_dates_years(bond) - расчет числа полных лет для дисконтирования будущих купонов к дате анализа
    cr_rf_rate(bond) - список безрисковых ставок с КБД
    cr_prices(bond) - выгрузка и создание массива исторических цен
    cr_rets(bond) - расчет исторических логарифмических доходностей
    cr_windows(bond) - нарезка истории на окна и определение длины каждого окна
    cr_hist_yields(bond) - обрезка исторических данных для расчета исторических купонов и параметров облигации
    bond.hist(bond.yields, bond.start_obs, bond.ansys_date) - расчет исторических купонов и параметров облигации, которые будут передаваться в функцию расчета будущих денежных потоков в качестве точки отсчета 
    run_MC(bond) - расчет Монте-Карло
    calc_fair_price(bond) - расчет справедливой стоимости
    fair_price_distr(bond) - Расчет персентилей распределения справедливой цены
    fair_price_graph(bond) - Построение графика распределения справедливой цены
    ytm_distr(bond) - Расчет персентилей распределения доходности к погашению
    ytm_graph(bond) - Построение графика распределения доходности к погашению
    save_results(bond) - сохранение результатов в папке Output
    '''
    #if len(csv_path_wid.value)==0:
    print(bond.tic_curr)
    cr_prices(bond)
    if print_output==True:
        print('Закончена работа функции cr_prices()\nПроверьте, что:\nвыгрузились цены по всем базовым активам,\nисторических значений достаточно для анализа, \nнет цен после даты анализа\n',bond.prices,'\n')
    if print_output==True: check_prices(bond)
    cr_rets(bond)
    if print_output==True: 
        print('Закончена работа функции cr_rets()\nДоходности базовых активов\n',bond.rets,'\n')
    cr_calendar(bond)
    if print_output==True: 
        print('Закончена работа функции cr_calendar()\nСоздан календарь рабочих дней\nПроверьте, что в календаре нет выходных\n',bond.F_calendar,'\n')
    cr_days_to_generate(bond)
    if print_output==True: 
        print('Закончена работа функции cr_days_to_generate()\nПроверьте корректность расчета следующих атрибутов sp.bond:\n',
               'MaxOf2 ', bond.MaxOf2,' - последняя дата, на которую требуются прогнозные цены активов\n',
               'days_to_generate ', bond.days_to_generate,' - число дней, на которые требуются прогнозные цены активов\n', 
             )       
    cr_dates_years(bond)
    if print_output==True: 
        print('Закончена работа функции cr_dates_years()\nПроверьте корректность полных лет до дат выплат купонов (для дисконтирования, на исторические отрицательные значения не нужно обращать внимание)\n',bond.dates_years,'\n')
    cr_rf_rate(bond) #cr_rates(bond) Изменено!!!!!
    cr_rates(bond)
    if print_output==True: 
        print('Закончена работа функции cr_rf_rate()\nПроверьте, что безрисковые ставки выгрузились корректно для каждой даты выплаты купонов\n',bond.rf_rate,'\n')
    cr_windows(bond)
    if print_output==True: print('Закончена работа функции cr_windows()\nДлина каждого окна получилась ',bond.window,'\nНачальные дни окон для нарезки исторических данных\n',bond.windows_start,'\n')
    cr_hist_yields(bond)
    if print_output==True: 
        print('Закончена работа функции cr_hist_yields()\nДоходности базовых активов с даты выпуска по дату анализа\n',bond.yields,'\n')
        print('Запущен метод объекта облигации по расчету исторических параметров bond.hist(bond.yields, bond.start_obs, bond.ansys_date)\nНеобходимо проверить корректность рассчитываемых значений\n')
    bond.hist(bond.yields, bond.start_obs, bond.ansys_date)
    model.run_MC(bond)
    if model.FV == True:
        bond.calc_fair_price()
        print('Средние потоки по облигации', bond.opt_payoff+bond.fix_cf)
        print('Средневзвешенная по сценариям стоимость опциона -', bond.opt_price)
        print('Стоимость фиксированного потока по облигации -', bond.nominal_price)
        print('Средневзвешенная по сценариям справедливая стоимость облигации -', bond.fair_price)
        bond.fair_price_distr()
        print('Доверительный интервал для полученной справедливой стоимости:', bond.interval)
        #fair_price_graph(bond)
    if model.Ret == True:
        ytm_distr(bond)
        ytm_graph(bond)
    save_results(bond)
    

def fair_price_graph(obj):
    '''
    Построение графика распределения справедливой цены
    '''
    obj.fair_g=obj.fair.iloc[1:].plot(figsize=(10,8),lw=2, marker='', markersize=10, title='Распределение справедливой стоимости '+obj.bond_name+'\n')
    plt.legend(facecolor='white', framealpha=1)
    obj.fair_g.set_xlabel("Вероятность формирования справедливой стоимости не ниже Х руб.")
    obj.fair_g.set_ylabel("Справедливая стоимость")
    str_fair='Fair price: ' + str(np.round(obj.fair_price,2))+' RUB'
    up=(max(obj.fair.iloc[1:,0])-min(obj.fair.iloc[1:,0]))/50
    obj.fair_g.text(0.78, obj.fair_price+up, str_fair, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3})
    plt.xticks(np.arange(min(obj.fair.index), max(obj.fair.index)+0.05, 0.05))
    obj.fair_g.set_xticklabels(['{:,.0%}'.format(x) for x in obj.fair_g.get_xticks()])
    obj.fair_g.text(0.75, obj.fair_g.get_ylim()[1]+up, 'Дата анализа: '+obj.analysis_date)
    plt.show()
    
def ytm_distr(obj):
    '''
    Расчет персентилей распределения доходности к погашению
    '''
    obj.cfs_ytm_matrix = [obj.cfs_ytm[i:i + len(obj.cf_sim)] for i in range(0, len(obj.cfs_ytm), len(obj.cf_sim))]
    obj.ytms = []
    for k in range(len(obj.cfs_ytm_matrix)):      # Добавляем фиксированные потоки
        for n in range(len(obj.fix_cf)):
            obj.cfs_ytm_matrix[k][n] += obj.fix_cf[n]
    for i in range(len(obj.cfs_ytm_matrix)):      # Добавляем стоимость облигации в начало денежного потока
        if obj.YTM_method == 'Fair':
            obj.cfs_ytm_matrix[i].insert(0,-obj.fair_price)
        else:
            obj.cfs_ytm_matrix[i].insert(0,-obj.dirty_price)
    for cf in obj.cfs_ytm_matrix:                  # Расчет доходности к погашению
        #print('cf\n',cf)
        #print(obj.dates_years)
        curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=0.10)
        if curr_ytm == 0.1:
            curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=-0.2)#
            if curr_ytm == -0.2:
                curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=-0.3)#
                if curr_ytm == -0.3:
                    curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=-0.4)
                    if curr_ytm == -0.4:
                        curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=-0.5)#
                        if curr_ytm == -0.5:
                            curr_ytm=irr(cfs=cf, yrs=obj.dates_years, x0=-1)
                            #if curr_ytm == -1: print('YTM is not found')
        obj.ytms.append(curr_ytm)
        #print('ytm\n',obj.ytms[-1])
    #Рассчитываем персентили
    obj.perc=pd.DataFrame(data=np.round(np.percentile(obj.ytms,np.linspace(100,0,101)),6),columns=['YTMs'], index=[i/100 for i in range(101)])
    #для БД - конвертация в словарь
    obj.ytm_perc_db = obj.perc['YTMs'].to_dict()
    
    obj.perc['Median YTM']=np.median(obj.ytms)
    obj.perc['Mean YTM']=np.mean(obj.ytms)
    ##
    obj.mean_ytm_db = dict(enumerate(obj.perc['Mean YTM'][0].flatten(), 1))
    obj.perc.index.name="YTM percentiles for "+obj.bond_name

def ytm_graph(obj):
    '''
    Построение графика распределения доходности к погашению
    '''
    obj.ytm_g=obj.perc.iloc[1:].plot(figsize=(10,8),
                                     lw=2, 
                                     marker='', 
                                     markersize=10, 
                                     title='Распределение доходностей к погашению\n' + obj.bond_name + '\n')
    plt.legend(facecolor='white', framealpha=1)
    
    obj.ytm_g.set_xlabel("Вероятность получить доходность не ниже Х%")
    obj.ytm_g.set_ylabel("Доходность к погашению при цене " + str(np.round(-obj.cfs_ytm_matrix[0][0],2)))
    plt.xticks(np.arange(min(obj.perc.index), max(obj.perc.index)+0.05, 0.05))
    y_min=min(obj.perc.iloc[1:,0])
    y_max=max(obj.perc.iloc[1:,0])
    
    if y_max - y_min < 0.5:
        plt.yticks(np.arange(y_min, y_max + 0.01, 0.01))
    else:
        y_min=np.floor(y_min*10)/10
        y_max=np.ceil(y_max*10)/10
        plt.yticks(np.arange(y_min, y_max + 0.05, 0.05))
    vals = obj.ytm_g.get_yticks()
    obj.ytm_g.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    vals = obj.ytm_g.get_xticks()
    obj.ytm_g.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    str_ytm='Mean YTM: ' + str(np.round(np.mean(obj.ytms)*100,2))+'%'
    up=(max(obj.perc.iloc[1:,0])-min(obj.perc.iloc[1:,0]))/50
    obj.ytm_g.text(0.82, np.mean(obj.ytms)+up, str_ytm, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3})
    obj.ytm_g.text(0.75, obj.ytm_g.get_ylim()[1]+up, 'Код: '+obj.ISIN+'\nДата анализа: '+obj.analysis_date)
    plt.show()

def save_results(obj):
    '''
    Сохранение картинок с графиками распределений и источника их построения в Excel в папку Output, находящуюся в папке с моделью.
    '''
    #fig=obj.fair_g.get_figure()
    #fig.savefig("Output/output "+obj.bond_name+ " Prices" + ' date '+obj.analysis_date+".png",dpi=500)
    if model.Ret == True:
        fig=obj.ytm_g.get_figure()
        fig.savefig("Output/output "+obj.bond_name+'(code ' +obj.ISIN + ')'+ " YTM" + ' date '+obj.analysis_date+' price '+str(np.round(-obj.cfs_ytm_matrix[0][0],2))+".png",dpi=500)
        obj.avg_cf = pd.DataFrame(obj.opt_payoff + obj.fix_cf, columns = ['AVG_cf'])
        obj.dates_years_out = pd.DataFrame(data = obj.dates_years[1:],columns = ['Dates_years'])
        obj.df = pd.concat([obj.dates_years_out, obj.avg_cf], axis = 1)
    if model.FV == True:
        obj.FV_results = pd.DataFrame({'Стоимость опциона':[bond.opt_price],
                                       'Стоимость фиксированного потока': [bond.nominal_price],
                                       'Справедливая стоимость': [bond.fair_price],
                                       'Доверительный интервал 95% (низ)': [bond.interval[0]],
                                       'Доверительный интервал 95% (верх)': [bond.interval[1]]}).T
        
    
    with pd.ExcelWriter("Output/output "+obj.bond_name+ '(code ' +obj.ISIN+ ')'+' date '+obj.analysis_date+'.xls') as writer:
        if model.Ret == True:
            obj.perc.to_excel(writer,sheet_name='YTMs for price '+str(np.round(-obj.cfs_ytm_matrix[0][0],2)))
            obj.df.to_excel(writer,sheet_name='AVG CFs')
        if model.FV == True:
            obj.FV_results.to_excel(writer,sheet_name='Fair Value')
    
attr_list=['bond_type', 'bond_name', 'ISIN', 'issue_date', 'coupon_dates', \
           'start_observation' , 'observations' , 'tickers', 'base_source', \
           'data_source', 'include_dividends', 'dateformat' , 'principal', \
           'fix_cf', 'dirty_price', 'risk_spread', 'YTM_method']                       