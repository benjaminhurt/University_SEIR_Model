import pandas as pd
from datetime import datetime,timedelta
import sys

def prep_case_data(obsdt):
    ### Case data
    strt_dt='2020-08-01' # saturday
    # strt_dt='2020-07-29' # wednesday
    # strt_dt='2021-01-01'

    # st0 = datetime.strptime('2020-08-01','%Y-%m-%d')
    # str_dates_new=[(st0+timedelta(days=x)).strftime('%Y-%m-%d') for x in range(365)]
    # dstr_dates_new=[datetime.strptime(d,"%Y-%m-%d").date() for d in str_dates_new]
    # print(dstr_dates_new)
    # dictuniv=pd.read_csv('../input/univdict.csv')

    ccdf=pd.read_csv('../input/university_csv/VA_univ_cum_cases_bh_2021-03-16.csv')
    ccdf.date=pd.to_datetime(ccdf.date,infer_datetime_format=True)
    ccdf=ccdf.dropna(subset=['date'])
    ccdf.date=ccdf.date.apply(lambda x: x.strftime('%Y-%m-%d'))
    ccdf=ccdf.set_index('date')
    ccdf=ccdf.loc[strt_dt:obsdt]
    ccdf.loc[:obsdt]=ccdf.loc[:obsdt].fillna(0)
    ccdf=ccdf.dropna()
    udf=ccdf.diff().fillna(0)

    udf=udf.rename(columns={'UIUC*':'UIUC'})
    udf=udf.rename(columns={'James Madison University':'JMU','Virginia Tech':'VT','Old dominion':'ODU','George Mason':'GMU', 'Virginia Commonwealth University' : "VCU"})
    udf=udf.rename(columns={'Arizona State University':'ASU','UTAus':'UTAUS'})

    for i in range(7):
        ddt=(datetime.strptime(strt_dt,'%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
        udf.loc[ddt,:]=0

    udf=udf.sort_index()
    udf[udf<0]=0

    rudf=udf.rolling(7).mean().fillna(0)
    rudf=rudf.loc[strt_dt:]
    udf=udf.loc[strt_dt:]

    # udf.to_csv('../input/udf_' + obsdt + '.csv')
    # rudf.to_csv('../input/rudf_' + obsdt + '.csv')

    return rudf