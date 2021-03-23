
import pandas as pd
import numpy as np
import argparse
from datetime import datetime,timedelta
from scipy.signal import savgol_filter
from scipy.optimize import minimize, Bounds
import sys
sys.path.append('/project/biocomplexity/nssac/PatchSim/')
import patchsim as sim
from univ_script_util import get_err_old, run_MC, get_beta, read_config_file
from prep_script_files import prep_case_data


ap = argparse.ArgumentParser()
ap.add_argument('--obsdt', type=str, default='2021-03-16', help="Last ground truth data date")
ap.add_argument('--univ_config_loc', type=str, default='../input/cfg_univ_model', help="University config file location")
ap.add_argument('--sim_config_loc', type=str, default='../input/single/cfg_base_uva', help="Patchsim config location")
ap.add_argument('--school', type=str, default='JMU', help="School to find cases for")
ap.add_argument('--scen', type=float, default=1.0, help="Beta scenario")
args = ap.parse_args()
print('--------Parameters--------')
print(args)

# Load university config file
# config_file_loc = '../input/cfg_univ_model'
univ_config = read_config_file(args.univ_config_loc)
univ_config['freq'] = int(univ_config['freq'])
univ_config['delay'] = int(univ_config['delay'])
print('univ model configs: \n', univ_config)

rudf = prep_case_data(args.obsdt)
configs = sim.read_config(args.sim_config_loc)

# ### College Update

def univ_projection(rudf, school, scen, sim_config, univ_configs):  # ['GMU', VT','UNC','UND','UIUC','UTAUS','OSU','UArizona','UVA']:

    horizon = len(rudf)
    siglen = len(rudf)
    full_len = siglen + 14
    beta = univ_configs['gamma'] * univ_configs['R0']
    init_R0_wk = np.ones(horizon // univ_configs['freq'] + 1)

    sim_config['InfectionRate'] = univ_configs['alpha']
    sim_config['RecoveryRate'] = univ_configs['gamma']
    sim_config['ExposureRate'] = beta

    ubetadf = pd.DataFrame()
    usimdf = pd.DataFrame()
    usimdf_lb = pd.DataFrame()
    usimdf_ub = pd.DataFrame()

    sim_config['Duration'] = horizon

    smooth_beta = np.zeros([1, horizon])
    smooth_R0 = np.zeros([1, horizon])

    k = 0
    # for target in ['perK']:#cnewdf.columns:
    target = school
    sim_config['PatchFile'] = '../input/single/pop_{}.txt'.format(school)
    sim_config['SeedFile'] = '../input/seed/{}seed_{}.txt'.format('UVA', 100)
    print('patchsim configs: \n', sim_config)
    print('for {}'.format(school))

    ### Prepare ground truth time series and initial R0
    # dates = [pd.Timestamp(x) for x in udf[target].index.values[:horizon]]
    st0 = datetime.strptime('2020-08-01','%Y-%m-%d')
    str_dates_new=[(st0+timedelta(days=x)).strftime('%Y-%m-%d') for x in range(365)]
    dstr_dates_new=[datetime.strptime(d,"%Y-%m-%d").date() for d in str_dates_new]
    ts_vec = rudf[target].values[:horizon]

    ### Run SLSQP with given R0 bounds
    R0_min = 0.5
    R0_max = 15
    print('optimization started for {}'.format(target))
    # res = minimize(get_err_old, init_R0_wk, args=(
    #     sim_config, univ_configs['freq'], horizon, univ_configs['gamma'], ts_vec, univ_configs['delay'],
    #     univ_configs['scale']), method='SLSQP', bounds=Bounds(R0_min, R0_max))
    try:
        res = minimize(get_err_old, init_R0_wk, args=(sim_config, univ_configs['freq'], horizon, univ_configs['gamma'], ts_vec, univ_configs['delay'], univ_configs['scale']), method='SLSQP', bounds=Bounds(R0_min, R0_max))
        print(target, '- Done')
    except:
        print('Something failed for {}'.format(target))
    #     continue

    ### Smooth the optimal Reff vector, get beta and produce simulated curve
    opt_R0_wk = res.x
    opt_beta = get_beta(opt_R0_wk, univ_configs['freq'], horizon, univ_configs['gamma'])

    # replace last 7 values of opt_beta with the average of the last 7 values of opt_beta
    opt_beta[0, -7:] = opt_beta[0, -7:].mean() * scen

    smooth_beta[k, :] = savgol_filter(opt_beta, window_length=7, polyorder=1)
    smooth_R0[k, :] = (smooth_beta[k] / univ_configs['gamma'])  # [:-delay]

    sim_config['Duration'] = 250
    patch_df = sim.load_patch(sim_config)
    params = sim.load_params(sim_config, patch_df)

    smooth_beta_ext = np.zeros([1, sim_config['Duration']])
    smooth_beta_ext[0, :horizon] = smooth_beta[0, :horizon]
    smooth_beta_ext[0, horizon:] = np.repeat(smooth_beta[0, -1], sim_config['Duration'] - horizon)

    col_sc = school
    ubetadf[col_sc] = smooth_beta_ext[0, :]
    ubetadf.loc[siglen:, col_sc] = smooth_beta_ext[0, siglen:]  # /100*smooth_beta_ext[0,siglen:full_len]

    beta_eff = ubetadf[col_sc].values[:full_len]  # adpdf.loc['51003'].values

    beta_ext = beta_eff
    temp, mn_temp, sd_temp = run_MC(beta_ext, sim_config, full_len, siglen, univ_configs['delay'], univ_configs['scale'], mc_len=200, stdv=0.25)
    usimdf[col_sc] = mn_temp
    usimdf_lb[col_sc] = mn_temp - 1.96 * sd_temp
    usimdf_ub[col_sc] = mn_temp + 1.96 * sd_temp
    # cs, cs_lb, cs_ub = get_case_ct(mn_temp, sd_temp)
    # ucsdf.loc[0, col_sc] = cs
    # ucsdf.loc[0, col_sc + '_lb'] = cs_lb
    # ucsdf.loc[0, col_sc + '_ub'] = cs_ub

    usimdf_lb[usimdf_lb < 0] = 0
    usimdf_ub[usimdf_ub < 0] = 0

    # # prepare results for saving (just predictions)
    usimdf_multi_plot = usimdf.loc[full_len-14:full_len-1]
    usimdf_lb_multi_plot = usimdf_lb.loc[full_len-14:full_len-1]
    usimdf_ub_multi_plot = usimdf_ub.loc[full_len-14:full_len-1]
    # change prediction index to be dates
    usimdf_multi_plot.index = dstr_dates_new[full_len-14:full_len-1]
    usimdf_lb_multi_plot.index = dstr_dates_new[full_len-14:full_len-1]
    usimdf_ub_multi_plot.index = dstr_dates_new[full_len-14:full_len-1]

    return usimdf_multi_plot, usimdf_lb_multi_plot, usimdf_ub_multi_plot

# univ_list = ['UVA', 'JMU'] #['UVA', 'JMU', 'VT', 'VCU']
usimdf_multi_plot, usimdf_lb_multi_plot, usimdf_ub_multi_plot = univ_projection(rudf, args.school, args.scen, configs, univ_config)
# print(usimdf_multi_plot)

# save results down
usimdf_multi_plot.to_csv('../output/pred plots/univ_pred_{}_{}_{}.csv'.format(args.school, args.obsdt, args.scen))
usimdf_lb_multi_plot.to_csv('../output/pred plots/univ_pred_lb_{}_{}_{}.csv'.format(args.school, args.obsdt, args.scen))
usimdf_ub_multi_plot.to_csv('../output/pred plots/univ_pred_ub_{}_{}_{}.csv'.format(args.school, args.obsdt, args.scen))
