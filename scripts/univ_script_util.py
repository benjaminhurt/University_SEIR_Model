import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from scipy.signal import savgol_filter
from scipy.optimize import minimize, Bounds
import sys
sys.path.append('/project/biocomplexity/nssac/PatchSim/')
import patchsim as sim
import matplotlib.pyplot as plt

import pdb
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


# %%

def read_config_file(config_file_loc):
    univ_config_df = pd.read_csv(config_file_loc, delimiter="=", names=["key", "val"])
    univ_config = dict(zip(univ_config_df.key, univ_config_df.val))
    return univ_config

def df_shift_scale(df, delay, scale):
    return df.reindex(columns=range(0, df.columns.max() + delay)).fillna(0.0).astype(float).shift(periods=delay,
                                                                                                  axis='columns').fillna(
        0.0) * scale

def get_sim(configs, beta, delay, scale):
    patch_df = sim.load_patch(configs)
    params = sim.load_params(configs, patch_df)
    params['beta'] = beta
    df = df_shift_scale(sim.run_disease_simulation(configs, params=params, return_epi=True), delay, scale)
    return (df.loc['A'].values)


def get_sim_intv(beta, configs, delay, scale):
    patch_df = sim.load_patch(configs)
    params = sim.load_params(configs, patch_df)
    seeds = sim.load_seed(configs, params, patch_df)
    Theta = sim.load_Theta(configs, patch_df)
    vaxs = sim.load_vax(configs, params, patch_df)
    params['beta'] = beta
    improve_testing = Improve_Testing()
    df = df_shift_scale(
        sim.run_disease_simulation(configs, patch_df, params, Theta, seeds, vaxs, return_epi=True, write_epi=False,
                                   log_to_file=False, intervene_step=improve_testing), delay, scale)
    return (df.loc['A'].values)


def run_patch(configs, patch_df, params, Theta, seeds, vaxs):
    improve_testing = Improve_Testing()
    df = sim.run_disease_simulation(configs, patch_df, params, Theta, seeds, vaxs, return_epi=True, write_epi=False,
                                    log_to_file=False, intervene_step=improve_testing)
    df.columns = range(len(df.columns))
    return df


def get_del_beta(smooth_beta, day_delay):
    del_smooth_beta = np.zeros(smooth_beta.shape)
    del_smooth_beta[0, :] = np.append(np.append(smooth_beta[0][:13], np.repeat(smooth_beta[0][13], day_delay)),
                                      smooth_beta[0][13:(len(smooth_beta) - (day_delay + 1))])
    return del_smooth_beta


def get_beta(R0_wk, freq, horizon, gamma):
    beta = np.repeat(R0_wk, freq)[:horizon].reshape(1, horizon) * gamma
    return (beta)


# def get_err(R0_wk, gt, target, freq, horizon, gamma, delay, scale):
#     ts_vec = gt[target].values
#     beta = get_beta(R0_wk, freq, horizon, gamma)
#     sim_vec = get_sim(beta, delay, scale)[:horizon]
#     err_vec = sim_vec - ts_vec
#     rmse = np.sqrt(np.sum(err_vec ** 2))
#     return (rmse)


def get_err_old(R0_wk, configs, freq, horizon, gamma, ts_vec, delay, scale):
    beta = get_beta(R0_wk, freq, horizon, gamma)
    sim_vec = get_sim(configs, beta, delay, scale)[:horizon]
    err_vec = sim_vec - ts_vec
    rmse = np.sqrt(np.sum(err_vec ** 2))
    return (rmse)

def get_ratio_CI(del_sim_vec, sim_vec):
    ratio_del = np.zeros(del_sim_vec.shape[0])
    for i in range(del_sim_vec.shape[0]):
        ratio_del[i] = del_sim_vec[i, -1] / sim_vec[i, -1]
    mn_ratio_del = ratio_del.mean()
    std_ratio_del = ratio_del.std()
    ratio_lci = mn_ratio_del - 1.96 * std_ratio_del
    ratio_uci = mn_ratio_del + 1.96 * std_ratio_del
    return mn_ratio_del, ratio_lci, ratio_uci


def get_day_num(date, str_dates_new):
    curr_date = datetime.strptime(date, '%Y-%m-%d')
    st0_date = datetime.strptime(str_dates_new[0], '%Y-%m-%d')
    num = curr_date - st0_date
    return num.days


def run_MC(smooth_beta, configs, full_len, horizon, delay, scale, mc_len=200, stdv=0.025):
    #     configs = sim.read_config('../input/single/{}'.format(scen[sc]))
    #     print(configs)
    configs['Duration'] = full_len
    smooth_beta = smooth_beta.reshape(1, configs['Duration'])
    sim_vec = np.zeros([mc_len, full_len - 1])
    mn_sim_vec = np.zeros([full_len - 1])
    sd_sim_vec = np.zeros([full_len - 1])
    for i in range(mc_len):
        old_hrzn = horizon - 8
        shp = smooth_beta[:, old_hrzn:].shape
        rand_temp = np.zeros(smooth_beta.shape)
        rand_temp[:, old_hrzn:] = stdv * np.multiply(np.linspace(0, 1, shp[1]), np.random.randn(shp[0], shp[1]))
        beta_temp = smooth_beta + rand_temp
        sim_vec[i, :] = get_sim(configs, beta_temp, delay, scale)[:-7]  # smooth_beta+stdv*np.random.randn(full_len)
        mn_sim_vec = sim_vec[:, :].mean(axis=0)
        sd_sim_vec = sim_vec[:, :].std(axis=0)
    return sim_vec, mn_sim_vec, sd_sim_vec


def run_MC_intv(smooth_beta, configs, full_len, horizon, delay, scale, mc_len=200, stdv=0.025):
    #     configs = sim.read_config('../input/single/{}'.format(scen[sc]))
    #     print(configs)
    configs['Duration'] = full_len
    smooth_beta = smooth_beta.reshape(1, configs['Duration'])
    sim_vec = np.zeros([mc_len, full_len - 1])
    mn_sim_vec = np.zeros([full_len - 1])
    sd_sim_vec = np.zeros([full_len - 1])
    for i in range(mc_len):
        old_hrzn = horizon - 8
        shp = smooth_beta[:, old_hrzn:].shape
        rand_temp = np.zeros(smooth_beta.shape)
        rand_temp[:, old_hrzn:] = stdv * np.multiply(np.linspace(0, 1, shp[1]), np.random.randn(shp[0], shp[1]))
        beta_temp = smooth_beta + rand_temp
        sim_vec[i, :] = get_sim_intv(beta_temp, configs, delay, scale)[:-7]  # smooth_beta+stdv*np.random.randn(full_len)
        mn_sim_vec = sim_vec[:, :].mean(axis=0)
        sd_sim_vec = sim_vec[:, :].std(axis=0)
    return sim_vec, mn_sim_vec, sd_sim_vec

# def get_case_cnty(mn_temp,sd_temp,cnty):
#     csct=mn_temp[horizon:].cumsum()[-1]*popdict[cnty]/100000
#     csct_lb=(mn_temp[horizon:]-1.96*sd_temp[-1]).cumsum()[-1]*popdict[cnty]/100000
#     csct_ub=(mn_temp[horizon:]+1.96*sd_temp[-1]).cumsum()[-1]*popdict[cnty]/100000
#     return csct,csct_lb,csct_ub

def get_case_ct(mn_temp, sd_temp):
    csct = mn_temp.cumsum()[-1]  # *pop/100000
    csct_lb = (mn_temp - 1.96 * sd_temp).cumsum()[-1]  # *pop/100000
    csct_ub = (mn_temp + 1.96 * sd_temp).cumsum()[-1]  # *pop/100000
    return csct, csct_lb, csct_ub

def get_dbt(cdf):
    tdf = pd.DataFrame(index=cdf.index)
    x = cdf.values
    p = np.zeros(len(x))
    for i in range(1, len(x)):
        p[i] = np.log(2) / np.log(x[i] / x[i - 1])
    col = cdf.columns[0]
    tdf.loc[cdf.index, col] = p
    return tdf


class Improve_Testing:
    def __init__(self):
        self.original_gamma = None
        self.reduced_gamma = None

    def __call__(self, configs, patch_df, params, Theta, seeds, vaxs, t, State_Array):
        if self.original_gamma is None:
            self.original_gamma = params["gamma"]

        if t > configs['LastDayNum']:
            self.reduced_gamma = self.original_gamma + configs['Freq'] * configs['Snstvty']
            params["gamma"] = self.reduced_gamma
        else:
            params["gamma"] = self.original_gamma