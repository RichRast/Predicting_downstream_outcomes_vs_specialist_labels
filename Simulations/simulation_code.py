import numpy as np
import copy
from scipy.special import expit
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns

CORRELATION_DIFF_LATEX_NAME = '$corr(\hat p_{Y_{T=1}}, p_Y) - corr(\hat p_T, p_Y)$'

def fit_estimators(X, T, D):
    """
    Fit the estimators we have been discussing given X, T, D. 
    Note these don't currently use a test set, just fit on the original data. 
    They also just fit logistic regression. 
    Maybe okay for toy models, probably not great in general. 
    """
    d = pd.DataFrame({'T':T, 'D':D})
    assert len(X.shape) == 2
    assert len(X) == len(d)
    assert (X[:, 0] == 1).all() # intercept columns. 
    X_df = pd.DataFrame(X, columns=['X%i' % i for i in range(X.shape[1])])
    d = pd.concat([d, X_df], axis=1)
    X_str = '+'.join(X_df.columns) + '-1' # no intercept column, included automatically in X. 
    d['T_and_D'] = d['T'] * d['D']
    estimators = {}
    estimators['p(D=1|T=1, X)'] = sm.Logit.from_formula('D ~ ' + X_str, data=d.loc[d['T'] == 1]).fit(disp=0).predict(d)
    estimators['p(D=1, T=1|X)'] = sm.Logit.from_formula('T_and_D ~ ' + X_str, data=d).fit(disp=0).predict(d)
    estimators['p(T=1|X)'] = sm.Logit.from_formula('T ~ ' + X_str, data=d).fit(disp=0).predict(d)
    estimators['p(T=1|X) * p(D=1|T=1, X)'] = estimators['p(T=1|X)'] * estimators['p(D=1|T=1, X)']
    # IPW estimator. Less sure about this one, did not include for now. 
    #d['IPW_weight'] = 1 / estimators['p(T=1|X)'].values
    #resampled_d = d.loc[d['T'] == 1].sample(weights='IPW_weight', replace=True, n=len(d))
    #estimators['IPW p(D=1|T=1, X)'] = sm.Logit.from_formula('D ~ ' + X_str, data=resampled_d).fit(disp=0).predict(d)
    return pd.DataFrame(estimators)

def generate_simulated_data_without_using_Z(N, beta_XT, beta_XD1, intercept_D0, beta_D0, make_plot=True, warn_if_out_of_range=False):
    """
    Generate N datapoints from following model
    
    X ~ Multivariate_Normal(0, I) except for the first column, which is all ones (this allows us to control mean of T + D). 
    p(T=1|X) = sigmoid(X * beta_XT)
    p(D=1|T=1, X) = sigmoid(X * beta_XD1)
    p(D=1|T=0, X) = p(D=1|T=1, X) * sigmoid(X * beta_D0) + intercept_D0
    
    intercept_D0 <= 0 and 0 <= sigmoid(X * beta_D0) <= 1 so p(D=1|T=0, X) <= p(D=1|T=1, X)
    Note that p(D=1|T=0, X) is not guaranteed to be >= 0 because of the intercept term 
    The way the code deals with this currently is by throwing a warning if warn_if_out_of_range is True and clipping, which maybe isn't ideal
    But I wanted to include the intercept because it is the case we actually proved something about. 
    """
    assert intercept_D0 <= 0
    M = len(beta_XT)
    X = np.random.randn(N, M) 
    X[:, 0] = 1
    p_T = expit(np.dot(X, beta_XT)) # p(T=1|X)
    p_D_T1 = expit(np.dot(X, beta_XD1)) # p(D=1|T=1, X)
    if beta_D0 is not None:
        p_D_T0 = p_D_T1 * expit(np.dot(X, beta_D0))
    else:
        p_D_T0 = p_D_T1.copy()
    p_D_T0 = p_D_T0 + intercept_D0 # p(D=1|T=0, X)
    p_D_T0_out_of_range_frac = 0
    if not ((p_D_T0 >= 0).all() and (p_D_T0 <= 1).all()):
        p_D_T0_out_of_range_frac = ((p_D_T0 < 0) | (p_D_T0 > 1)).mean()
        if warn_if_out_of_range:
            print("WARNING: fraction %2.3f of entries of p_D_T0 are out of range: min %2.3f, max %2.3f" % 
              (p_D_T0_out_of_range_frac, p_D_T0.min(), p_D_T0.max()))
        p_D_T0 = np.clip(p_D_T0, 0, 1)
    T = (np.random.random(N,) < p_T) * 1.
    D = np.empty(T.shape)
    D[T == 1] = np.random.random((T == 1).sum()) < p_D_T1[T == 1]
    D[T == 0] = np.random.random((T == 0).sum()) < p_D_T0[T == 0]
    p_D = p_T * p_D_T1 + (1 - p_T) * p_D_T0
    if make_plot:
        sns.pairplot(pd.DataFrame({'p_T':p_T, 'p_D_T1':p_D_T1, 'p_D_T0':p_D_T0, 'p_D':p_D}))

    # sanity checks on all output
    for vector in [p_T, p_D, p_D_T1, p_D_T0, p_D_T1 - p_D_T0]:
        assert ((vector < 0) | (vector > 1)).sum() == 0
    assert ((D == 1) | (D == 0)).all()
    assert ((T == 1) | (T == 0)).all() 
    assert D.shape == T.shape == p_T.shape == p_D.shape == p_D_T1.shape == p_D_T0.shape == (N,)
        
    return {'X':X, 
            'D':D, 
            'T':T, 
            'ground truth p(T=1|X)':p_T,
            'ground truth p(D=1|X)':p_D, 
            'ground truth p(D=1|T=1, X)':p_D_T1, 
            'ground truth p(D=1|T=0, X)':p_D_T0, 
            'ground truth u(X)':p_D_T1 - p_D_T0, 
            'p_D_T0_out_of_range_frac':p_D_T0_out_of_range_frac}

def fit_param_set_and_print_results(params, verbose=True):
    """
    Return summary statistics from a simulation to use for large-scale simulation results. 
    """
    ground_truth = generate_simulated_data_without_using_Z(**params)
    estimators = fit_estimators(X=ground_truth['X'], 
                   T=ground_truth['T'], 
                   D=ground_truth['D'])
    if verbose:
        print("\n\nCorrelation between ground truth p(T=1|X) and p_hat(T=1|X): %2.3f" % 
              pearsonr(ground_truth['ground truth p(T=1|X)'], 
                       estimators['p(T=1|X)'])[0])
        print("Correlation between ground truth p(D=1|T=1, X) and p_hat(D=1|T=1, X): %2.3f" % 
              pearsonr(ground_truth['ground truth p(D=1|T=1, X)'], 
                       estimators['p(D=1|T=1, X)'])[0])
        print("Fraction of points with T = 1: %2.3f" % ground_truth['T'].mean())

        print("Std of p(T=1|X): %2.3f; p(D=1|T=1, X): %2.3f; unobservables crossing point is %2.3f. For constant unobservables and large datasets, p(T=1|X) should better correlate with ground truth only if unobservables exceed this value." % 
              (ground_truth['ground truth p(T=1|X)'].std(), 
               ground_truth['ground truth p(D=1|T=1, X)'].std(), 
               ground_truth['ground truth p(D=1|T=1, X)'].std()/ground_truth['ground truth p(T=1|X)'].std()))

        print("p_hat(D=1|T=1, X) correlation with ground truth: %2.3f" % pearsonr(ground_truth['ground truth p(D=1|X)'], estimators['p(D=1|T=1, X)'])[0])
        print("p_hat(T=1|X) correlation with ground truth: %2.3f" % pearsonr(ground_truth['ground truth p(D=1|X)'], estimators['p(T=1|X)'])[0])
    # don't return the raw predictions, which get large; rather, return their means, sds, and correlations. 
    matrix_to_correlate = pd.DataFrame({'p(D=1|T=1, X)':ground_truth['ground truth p(D=1|T=1, X)'], 
                           'p(T=1|X)':ground_truth['ground truth p(T=1|X)'],
                           'p(D=1|X)':ground_truth['ground truth p(D=1|X)'], 
                           'u(X)':ground_truth['ground truth u(X)'],
                           'u(X) * p(T=0|X)':ground_truth['ground truth u(X)']*(1 - ground_truth['ground truth p(T=1|X)']),
                           'p_hat(D=1|T=1, X)':estimators['p(D=1|T=1, X)'], 
                           'p_hat(T=1|X)':estimators['p(T=1|X)']})
          
    return {'mean':matrix_to_correlate.mean(), 
            'std':matrix_to_correlate.std(), 
            'spearman_corr':matrix_to_correlate.corr(method='spearman'), 
            'pearson_corr':matrix_to_correlate.corr(method='pearson'), 
            'p_D_T0_out_of_range_frac':ground_truth['p_D_T0_out_of_range_frac']}

"""
Single functions of things we want to extract from simulation results. Functions should be self-explanatory, hopefully. 
"""
def corr_p_d_estimate_with_p_d_ground_truth(x):  # x-axis measure. Quality of estimation of p(D=1|T=1, X)
    return x['pearson_corr']['p_hat(D=1|T=1, X)']['p(D=1|T=1, X)']

def corr_p_t_estimate_with_p_t_ground_truth(x): # x-axis measure. Quality of estimation of p(T=1|X)
    return x['pearson_corr']['p_hat(T=1|X)']['p(T=1|X)']
    
def sd_p_d_over_sd_p_t(x): # x-axis measure. How much variation in p(D=1|T=1, X) relative to p(T|X)
    return x['std']['p(D=1|T=1, X)']/x['std']['p(T=1|X)']

def sd_p_d_over_sd_p_t0_times_u(x): # x-axis measure. How much variation in p(D=1|T=1, X) relative to p(T|X) scaled by unobservables
    return x['std']['p(D=1|T=1, X)']/x['std']['u(X) * p(T=0|X)']

def difference_in_correlations(x): # y-axis measure. Difference in estimator correlations with ground truth. 
    return x['pearson_corr']['p(D=1|X)']['p_hat(D=1|T=1, X)'] - x['pearson_corr']['p(D=1|X)']['p_hat(T=1|X)'] 

def D_better_correlated(x): # y-axis measure. Binary variable of which estimator is better-correlated with ground truth. 
    return x['pearson_corr']['p(D=1|X)']['p_hat(D=1|T=1, X)'] > x['pearson_corr']['p(D=1|X)']['p_hat(T=1|X)'] 

def p_D_T0_out_of_range_frac(x):
    return x['p_D_T0_out_of_range_frac']

def run_simulations_varying_parameter(default_param_set, 
                                      param_to_vary, 
                                      param_vals,
                                      n_trials_per_setting, 
                                      x_axis_names, 
                                      x_axis_fxns, 
                                      x_crossover_points,
                                      y_axis_names, 
                                      y_axis_fxns, 
                                      quantities_to_print_but_not_plot_names, 
                                      quantities_to_print_but_not_plot_fxns, 
                                      plot_filename_string=None):
    """
    Main function to run large-scale simulations. 
    Given a default parameter set, varies one param_to_vary at a time (holding all other parameters constant) by looping over param_vals. 
    For each trial runs n_trials_per_setting. 
    Saves all simulation results in a single dataframe and then plots various derived quantities from each simulation by looping over x_axis_fxns and y_axis_fxns. 
    x_axis_names and x_axis_fxns should be in corresponding order. 
    For example, for each simulation we might extract the std of p(D=1|T=1, X) to plot on the x-axis, and the difference in correlations with ground truth between p_hat(T=1|X) and p_hat(D=1|T=1, X) to plot on the y-axis. 
    We define a bunch of one-line functions to do this. 
    """
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":15,"axes.labelsize":15})   
    assert len(y_axis_names) == len(y_axis_fxns)
    assert len(x_axis_names) == len(x_axis_fxns)
    if x_crossover_points is not None: # do you want to plot vertical lines at some x-values. 
        assert len(x_axis_names) == len(x_crossover_points)
    assert len(quantities_to_print_but_not_plot_names) == len(quantities_to_print_but_not_plot_fxns)
    results = []
    for param_idx, val in enumerate(param_vals):
        param_set_to_use = copy.deepcopy(default_param_set)
        assert param_to_vary not in param_set_to_use
        param_set_to_use[param_to_vary] = val
        for i in range(n_trials_per_setting):
            if i == 0:
                print("Running trial %i/%i for parameter setting for %s %i/%i: %s" % 
                  (i + 1, n_trials_per_setting, param_to_vary, param_idx + 1, len(param_vals), val))
            results.append({'param_to_vary':param_to_vary, 
                            'trial':i, 
                            'param_val':val,
                            'param_idx':param_idx,
                            'results':fit_param_set_and_print_results(param_set_to_use, verbose=False)})
    results = pd.DataFrame(results)
    for i in range(len(quantities_to_print_but_not_plot_fxns)):
        results[quantities_to_print_but_not_plot_names[i]] = results['results'].map(quantities_to_print_but_not_plot_fxns[i]) # extract certain values from results column. 
    print("You wished to monitor the following quantities but not plot them")
    print(results.groupby('param_idx')[quantities_to_print_but_not_plot_names].agg(['mean', 'std']))

    print("actual results to be plotted")
    plot_idx = 0
    for i in range(len(y_axis_names)):
        for j in range(len(x_axis_names)):
            # make multiple sets of plots from a single simulation run for speed. Each time we plot a different quantity on x and y-axes for speed. 
            x_axis_name = x_axis_names[j]
            x_axis_fxn = x_axis_fxns[j]
            y_axis_name = y_axis_names[i]
            y_axis_fxn = y_axis_fxns[i]
            results[x_axis_name] = results['results'].map(x_axis_fxn)
            results[y_axis_name] = results['results'].map(y_axis_fxn)
            
            # lowess plot. Plot one point for each simulation run, and fit a lowess line. 
            plt.figure(figsize=[4, 4])
            sns.regplot(data=results, 
                       x=x_axis_name, 
                       y=y_axis_name, 
                       lowess=True,
                       scatter_kws={'alpha':0.1}, 
                       line_kws={'color':'black'})
            plt.xlim([results[x_axis_name].min(), results[x_axis_name].max()])

            if y_axis_name == CORRELATION_DIFF_LATEX_NAME: # define in variable because it's kind of lengthy. 
                ylim = results[y_axis_name].abs().max()
                plt.ylim([-ylim, ylim])
                plt.plot([results[x_axis_name].min(), results[x_axis_name].max()], 
                         [0, 0], 
                         color='black', 
                         linestyle='--')
                if x_crossover_points is not None:
                    plt.plot([x_crossover_points[j], x_crossover_points[j]], 
                         [-ylim, ylim], 
                         color='black', 
                         linestyle='--')
            if y_axis_name == 'Pr(D wins)':
                plt.ylim([0, 1])
                plt.plot([results[x_axis_name].min(), results[x_axis_name].max()], 
                         [0.5, 0.5], 
                         color='black', 
                         linestyle='--')
                if x_crossover_points is not None:
                    plt.plot([x_crossover_points[j], x_crossover_points[j]], 
                         [0, 1], 
                         color='black', 
                         linestyle='--')
            if (plot_filename_string is not None) and plot_idx == 0:
                plt.tight_layout()
                plt.savefig(plot_filename_string)
            
            
            # average across param settings. Basically this plots one point for each parameter setting. I found this made a less clean plot than just fitting LOWESS to all trials. 
            plt.figure()
            result_means = results.groupby('param_idx')[[x_axis_name, y_axis_name]].mean().reset_index()
            print(result_means)
            result_sds = results.groupby('param_idx')[[x_axis_name, y_axis_name]].std().reset_index()
            plt.errorbar(x=result_means[x_axis_name], 
                         xerr=result_sds[x_axis_name],
                         y=result_means[y_axis_name], 
                         yerr=result_sds[y_axis_name])
            plt.xlabel(x_axis_name)
            plt.ylabel(y_axis_name)
            plt.title('Average results')
            if y_axis_name == CORRELATION_DIFF_LATEX_NAME:
                plt.ylim([-2, 2])
                plt.plot([result_means[x_axis_name].min(), 
                          result_means[x_axis_name].max()], 
                         [0, 0], 
                         color='black', 
                         linestyle='--')
                if x_crossover_points is not None:
                    plt.plot([x_crossover_points[j], x_crossover_points[j]], 
                             [-2, 2], 
                             color='black', 
                             linestyle='--')
            if y_axis_name == 'Pr(D wins)':
                plt.ylim([0, 1])
                plt.plot([result_means[x_axis_name].min(), 
                          result_means[x_axis_name].max()], 
                         [0.5, 0.5], 
                         color='black', 
                         linestyle='--')
                if x_crossover_points is not None:
                    plt.plot([x_crossover_points[j], x_crossover_points[j]], 
                         [0, 1], 
                         color='black', 
                         linestyle='--')
                
            plt.xlim([result_means[x_axis_name].min(), result_means[x_axis_name].max()])
            plt.figure()
            plot_idx += 1
    
    return results 


