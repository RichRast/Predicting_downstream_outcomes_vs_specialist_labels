import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, average_precision_score,\
balanced_accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
import os
import os.path as osp
import pickle
import math
from functools import partial
import pdb

LABEL_DICT_FIRST_STAGE={'T': r'$p_T$', 'D|T': r'$p_{Y_{T=1}}$', 
        'D_and_T':r'$p_{Y=1,T=1}$', 'D|T_ipw':r'$p^{(IPW)}_{Y_{T=1}}$', 
        'D_pseudo':r'$p^{(pseudo)}_{Y}$', 'product_T_D_given_T':r'$p_T * p_{Y_{T=1}}$'}

LABEL_DICT_SECOND_STAGE={'T': r'$\hat{p}_T$', 'D|T': r'$\hat{p}_{Y_{T=1}}$', 
        'D_and_T':r'$\hat{p}_{Y=1,T=1}$', 'D|T_ipw':r'$\hat{p}^{(IPW)}_{Y_{T=1}}$', 
        'D_pseudo':r'$\hat{p}^{(pseudo)}_{Y}$', 'product_T_D_given_T':r'$\hat{p}_T * \hat{p}_{Y_{T=1}}$'}

DIFF_UNOBSERVABLES_PLOT=0.05

def trainEvalModel(model, train_X, train_y, test_X=None, test_y=None, model_descr="", calibrate=False, 
                   sample_weight=None, calibrate_method='sigmoid', cv=5):
    """
    Args:
        model: base estimator such as LGBMClassifier()
        train_X, train_y: train+val data
        test_X, test_y: hold-out test set
        model_descr: str description of model
        calibrate: If true, CalibratedClassifierCV uses 5 -fold cross validation to fit on training subset 
        and calibrate on validation subset 
        sample_weight: weights for each sample for the train+val subset
    Returns:
        model: instance of fitted model 
        test_proba: returns probabilties for the hold-out set
    """
    if calibrate:
        model=CalibratedClassifierCV(model, cv=cv, method=calibrate_method)
    if sample_weight is None:
        sample_weight=np.ones_like(train_y, dtype=np.dtype(float))
    assert sample_weight.dtype is np.dtype(float)
    # Since we are doing binary classification, the two asserts below check for y var
    assert train_y.dtype is np.dtype(int), f"{train_y.dtype}"
    assert (train_y.sum()<len(train_y)) and (train_y.sum()>0), f"{train_y.sum(), len(train_y)}"
    assert (train_X.dtype == np.dtype(float)) or (train_X.dtype==np.dtype('float64')), f"{train_X.dtype}"
    assert train_X.shape[0]==len(train_y)
    model.fit(train_X, train_y, sample_weight=sample_weight)
    
    if test_X is not None:
        assert test_y is not None, f"Cannot evaluate on holdout test set since test_y was not passed"
        test_proba = model.predict_proba(test_X)
        assert test_X.shape[0]==len(test_y)
        print(f"AUC score :{model_descr}: {roc_auc_score(test_y, test_proba[:,1]):.3f}")
        print(f"AUPR score :{model_descr}: {average_precision_score(test_y, test_proba[:,1]):.3f}")
        return model, test_proba

def hypertune(model, params_dict, train_X, train_y, cv=5):
    """
    Given a dict of params distributions, do a randomized search
    with k-fold cross validation
    Args:
        model: base classifier
        params_dict: dict with distribution of each param
        train_X, train_y: train+val data 
    Returns:
        best params dict after randomized search cv with k-folds
    """
    model=RandomizedSearchCV(model, params_dict, random_state=0, scoring='neg_log_loss', cv=cv)
    search = model.fit(train_X, train_y)
    print(f"cv results:{search.cv_results_}")
    print(f"best score: {search.best_score_}")
    print(f" best params:{search.best_params_}")
    return search.best_params_


def trainHardPseudo(clf_D_given_T, df, train_idxs, val_idxs, train_X, val_X, clf_D_pseudo, 
                    test_X_D_given_T, test_y_D_given_T, model_descr="", calibrate=True,  calibrate_method='sigmoid', train_y_D=None):
    """
    First form the pseudo labels from the model trained only on observed data.
    Use these labels to train a new model on all the data including the pseudolabels
    Args:
        
    """
    #assert the dtype of float for train_X and train_y and binary values for train_y
    train_y_D_predict = clf_D_given_T.predict(train_X)
    if df is None:
        assert train_y_D is not None, f" need to pass one of either dataframe or train_y_D"
        assert train_idxs is None, f" cannot pass both train_idxs and train_y_D"
        train_y_D_pseudo = train_y_D
    else:
        assert train_idxs is not None, f" need to pass one of either train_idxs or train_y_D"
        assert train_y_D is None, f" cannot pass both dataframe and train_y_D"
        train_y_D_pseudo = df.iloc[train_idxs]['D'].values
    assert np.isnan(train_y_D_pseudo).sum()>0, f" there were no nans/unobserved samples for outcome"
    train_y_D_pseudo[np.isnan(train_y_D_pseudo)] = train_y_D_predict[np.isnan(train_y_D_pseudo)]
    train_y_D_pseudo=train_y_D_pseudo.astype(int)
    assert train_y_D_pseudo.dtype is np.dtype(int), f"{train_y_D_pseudo.dtype}"
    np.testing.assert_array_equal (np.unique(train_y_D_pseudo), np.array([0, 1])), f"{np.unique(train_y_D_pseudo)}"

    if val_idxs is not None :
        assert val_X is not None, f"val_idxs are passed but Val_X is passed as None"
        val_y_D_predict = clf_D_given_T.predict(val_X).astype(int)
        val_y_D_pseudo = df.iloc[val_idxs]['D'].values
        val_y_D_pseudo[np.isnan(val_y_D_pseudo)]=val_y_D_predict[np.isnan(val_y_D_pseudo)]
        val_y_D_pseudo = val_y_D_pseudo.astype(int)
        assert val_y_D_pseudo.dtype is np.dtype(int), f"{val_y_D_pseudo.dtype}"
        np.testing.assert_array_equal (np.unique(val_y_D_pseudo), np.array([0, 1])), f"{np.unique(val_y_D_pseudo)}"

        train_cross_val_X = np.concatenate((train_X, val_X))
        train_cross_val_y_D_pseudo = np.concatenate((train_y_D_pseudo, val_y_D_pseudo))
    else:
        assert val_X is None, f"val_idxs are passed as None but Val_X is not None"
        print(f"no val data was passed")
        train_cross_val_X = train_X
        train_cross_val_y_D_pseudo = train_y_D_pseudo
    clf_D_pseudo, test_D_pseudo_proba = trainEvalModel(clf_D_pseudo, train_cross_val_X, train_cross_val_y_D_pseudo, test_X=test_X_D_given_T, 
                    test_y=test_y_D_given_T, model_descr=model_descr, calibrate=calibrate, calibrate_method=calibrate_method)
    return clf_D_pseudo, test_D_pseudo_proba


def plotCorr(models, dict_df_probs, title, corr_method='pearson', figsize=(15,9), top_adjust=0.85, title_en=True, second_stage=False, **kwargs):
    fig1 = plt.figure(figsize=figsize)
    savefig_path = kwargs.get('savefig_path')
    if title_en: plt.title(' '.join([corr_method, str(title)]), fontsize=15)
    
    for i, m in enumerate(models):
        df = dict_df_probs[m].copy()
        df = df.rename(columns=LABEL_DICT_SECOND_STAGE)
        corr_mat = df.corr(method=corr_method)
        assert np.all(np.linalg.eigvals(corr_mat)>0.0)
        mask = np.zeros_like(corr_mat, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        ax1 = plt.subplot(1, len(models), i+1)
        # if i!=len(models)-1 or not title_en:
        if not title_en:
            g1=sns.heatmap(corr_mat, mask=mask, cmap="Blues", annot=True, ax=ax1, cbar=True, annot_kws={"fontsize":30})
        else :
            g1=sns.heatmap(corr_mat, mask=mask, cmap="Blues", annot=True, ax=ax1, annot_kws={"fontsize":30})
            cbar = g1.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
        g1.set_ylabel('')
        g1.set_xlabel('')
        if not title_en:
            g1.set_yticks([])
        # g1.set_title(title)
        g1.tick_params(axis='both', which='major', labelsize=20, rotation=45)
    fig1.tight_layout()
    fig1.subplots_adjust(top=top_adjust)
    if savefig_path is not None:
        plt.savefig(f"{osp.join(savefig_path, ''.join([corr_method, '_second_stage_', str(second_stage), '_', str(title),'.pdf']))}")
    plt.show()
    plt.close()

def getPred_fromProb(prob, thresh=0.5):
    # if prob>=thresh:
    #     return 1.0
    # else:
    #     return 0.0
    return np.random.binomial(1,prob)
    
def getMetrics(labels, prob):
    PR = average_precision_score(labels, prob)
    auc_score = roc_auc_score(labels, prob)
    print(f"auc_score:{auc_score:.3f}")
    preds=np.vectorize(getPred_fromProb)(prob)
    balanced_accr = balanced_accuracy_score(labels, preds)
    cr = classification_report(labels, preds)
    return auc_score, PR, balanced_accr, cr

def loadModelData(m, t, df, test_idxs, clf_dict):
    """
    clf_dict is a dict with clf_dict[task][model]
    """
    assert m in ['LogisticRegression', 'LGBM', 'LGBM_w_feat', 'RF', 'XGB']
    assert t in ['T', 'D|T', 'D_and_T', 'D|T_ipw', 'D|T_ipw_pseudo', 'D|T_dr_pseudo', 'D_pseudo']
    clf=clf_dict[t][m]
    if t=='T':
        y=df.iloc[test_idxs]['T'].values
    elif t in ['D|T','D|T_ipw', 'D|T_ipw_pseudo', 'D|T_dr_pseudo', 'D_pseudo']:
        y=df.iloc[test_idxs]['D'].values
    elif t=='D_and_T':
        y=df.iloc[test_idxs]['D_and_T'].values
    return clf, y

def loadModelData_secondStage(m, t, test_second_stage, clf_dict):
    """
    clf_dict is a dict with clf_dict[task][model]
    """
    assert m in ['LogisticRegression', 'LGBM', 'LGBM_w_feat', 'RF', 'XGB']
    assert t in ['T', 'D|T', 'D_and_T', 'D|T_ipw', 'D|T_ipw_pseudo', 'D|T_dr_pseudo', 'D_pseudo']
    clf=clf_dict[t][m]
    if t=='T':
        y=test_second_stage['T']
    elif t in ['D|T','D|T_ipw', 'D|T_ipw_pseudo', 'D|T_dr_pseudo', 'D_pseudo']:
        y=test_second_stage['D']
    elif t=='D_and_T':
        y=test_second_stage['D_and_T']
    return clf, y

def plotCalibrationPlots(predProbs, y_true, modelName, taskName, ax1, ax2, n_bins=20):
    prob_true, prob_pred = calibration_curve(y_true, predProbs, n_bins=n_bins, strategy='quantile')
    ax1.plot([min(prob_true.min(), prob_pred.min()),max(prob_true.max(), prob_pred.max())],[min(prob_true.min(), prob_pred.min()), max(prob_true.max(), prob_pred.max())], linestyle='--', color='black')
    ax1.scatter(prob_pred, prob_true, s=15)
    ax1.set_xlabel('Mean predicted rate', fontsize=15)
    ax1.set_ylabel('Mean empirical rate', fontsize=15)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax2.hist(predProbs, bins=n_bins)
    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Counts')
    if taskName in ['Lactate_Measured', 'T']:
        estName=LABEL_DICT_SECOND_STAGE['T']
    elif taskName in ['Lactate_Above_Threshold', 'D|T']:
        estName=LABEL_DICT_SECOND_STAGE['D|T']
    elif taskName in ['Lactate_Measured_And_Disease', 'D_and_T']:
        estName=LABEL_DICT_SECOND_STAGE['D_and_T']
    elif taskName in ['Lactate_Above_Threshold_ipw', 'D|T_ipw']:
        estName=LABEL_DICT_SECOND_STAGE['D|T_ipw']
    elif taskName in ['D_pseudo']:
        estName=LABEL_DICT_SECOND_STAGE['D_pseudo']
    # ax1.set_title(f"{modelName}, task: {estName}")
    ax1.set_title(f"{estName}", fontsize=20)

def calibrate_by_bin(val_probs, val_y, test_probs, test_y, n_bins):
    assert val_probs.max() <= 1
    assert val_probs.min() >= 0
    assert test_probs.max() <= 1
    assert test_probs.min() >= 0
    
    val_df = pd.DataFrame({'probs':val_probs, 'ground_truth':val_y})
    val_df=val_df.dropna()

    # compute bin edges
    bin_percentile_cutoffs = np.linspace(0, 100, n_bins + 1)[1:-1]
    bin_value_cutoffs = np.percentile(val_df['probs'], q=bin_percentile_cutoffs)
    bin_value_cutoffs = [-0.001] + list(bin_value_cutoffs) + [1.001] # add the extreme edges
    
    # bin the validation data. 
   
    val_df['binned_val'] = pd.cut(val_df['probs'], bin_value_cutoffs).astype(str)
    val_grouped_d = val_df.groupby('binned_val')['ground_truth'].agg(['mean', 'size']).reset_index()
    # print(f"val sizes: {val_grouped_d['size']}")
    val_mapping = dict(zip(val_grouped_d['binned_val'], val_grouped_d['mean'])) # dictionary where each bin maps to mean value of outcome variable in that bin
    print(f"val max:{val_grouped_d['size'].max()}, val min:{val_grouped_d['size'].min()}")
    assert (val_grouped_d['size'].max() - val_grouped_d['size'].min() <= 3) # make sure bins are equal size
    
    # now apply mapping to test data. 
    test_df = pd.DataFrame({'probs':test_probs, 'ground_truth':test_y})
    test_df['binned_val'] = pd.cut(test_df['probs'], bin_value_cutoffs).astype(str) # first bin test data. 
    test_df['calibrated_preds'] = test_df['binned_val'].map(lambda x:val_mapping[x]) # then use same mapping. 

    # check that it worked. 
    print(f" number of bins in test :{len(set(test_df['calibrated_preds']))}")
    assert (len(set(test_df['calibrated_preds']))- n_bins <=3) # should have n_bins unique calibrated values. 
    df_to_plot = test_df.groupby('calibrated_preds')['ground_truth'].mean()
    print(df_to_plot)
    # print(f"test sizes: {test_df.groupby('calibrated_preds')['ground_truth'].size()}")
    plt.figure(figsize=[4, 4])
    plt.scatter(df_to_plot.index, df_to_plot.values)
    plt.xlabel("mean prediction in bin")
    plt.ylabel("mean outcome in bin")
    plt.plot([df_to_plot.index.min(), df_to_plot.index.max()], 
             [df_to_plot.index.min(), df_to_plot.index.max()], 
             color='black', 
             linestyle='--')
    
    return test_df['calibrated_preds'].values


def getCorr(models, tasks, X, clf_dict, df, test_idxs, dict_df_labels, dict_df_probs, dict_models, df_pp, probs_path, corr_method=stats.pearsonr, 
            calibrate=False, figsize1=(15,35), figsize2=(10,10), top_adjust=0.85, alpha=1.0, **kwargs):
    epsilon = kwargs.get('epsilon')
    test_second_stage = kwargs.get('test_second_stage')
    cal_second_stage = kwargs.get('cal_second_stage')
    cal_idxs = kwargs.get('cal_idxs')
    cal_X = kwargs.get('cal_X')

    for col, m in enumerate(models):
        dict_df_labels[m]= pd.DataFrame([])
        dict_df_probs[m]= pd.DataFrame([])
        dict_models[m]= {}
        fig1, ax1 = plt.subplots(nrows=2*len(tasks), figsize=figsize1)
        for row, t in enumerate(tasks):
            if test_second_stage is None:
                assert df is not None, f"need to pass dataframe for first stage"
                model , labels= loadModelData(m, t, df, test_idxs, clf_dict)
            else:
                model , labels= loadModelData_secondStage(m, t, test_second_stage, clf_dict)
            dict_models[m][str(t)]=model
            dict_df_labels[m][str(t)]=labels
            predict_probs = model.predict_proba(X)
            assert len(predict_probs)==len(X)
            assert predict_probs.shape[1]==2 # predicting binary class probbs, so index zero for class0, index 1 for class1
            probs=predict_probs[:,1]
            if (calibrate) and (cal_X is not None):
                print(f" calibrating by bins with cal data")
                val_probs = model.predict_proba(cal_X)[:,1]
                if test_second_stage is None:
                    assert df is not None, f"need to pass dataframe for first stage"
                    assert cal_idxs is not None, f" need to pass cal_idxs to extract labels from df"
                    _, val_labels=loadModelData(m, t, df, cal_idxs, clf_dict)
                else:
                    assert cal_second_stage is not None, f"Need to pass econd stage labels for cal data"
                    _, val_labels=loadModelData_secondStage(m, t, cal_second_stage, clf_dict)
                assert len(val_probs)==len(cal_X)
                probs = calibrate_by_bin(val_probs, val_labels, probs.copy(), labels.copy(), n_bins=20)
            dict_df_probs[m][str(t)]=probs
            probs = probs[~np.isnan(labels)]
            labels = labels[~np.isnan(labels)]
            auc_score, pr_score, balanced_acc = getMetrics(labels, probs)[:3]
            print(f"classification report for model {m}, task {t}: \n {getMetrics(labels, probs)[3]}")
            df_pp=df_pp.append({'AUC' :auc_score ,'PR':pr_score,'BalancedAcc':balanced_acc, 'modelName':m,'rowName':t},ignore_index=True)
            
            plotCalibrationPlots(probs, labels, m, t, ax1[2*row], ax1[(2*row)+1])
            #save probs
            saveFile(osp.join(probs_path, f"probs_{t}"), dict_df_probs[m][t], f"/probs.npy")
        fig1.tight_layout()
        fig1.subplots_adjust(top=top_adjust)
        plt.show()
        plt.close()
        # get the correlation
        T_probs = dict_df_probs[m]['T']
        D_given_T_probs = dict_df_probs[m]['D|T']
        D_and_T_probs = dict_df_probs[m]['D_and_T']
        corr_T=partial(corr_method, T_probs)
        corr_D_and_T=partial(corr_method, D_and_T_probs)
        print(f"T and D|T probs for model {m}: {corr_T(D_given_T_probs)}")
        print(f"T and D,T probs for model {m}: {corr_T(D_and_T_probs)}")
        print(f"D,T and D|T probs for model {m}: {corr_D_and_T(D_given_T_probs)}")
        fig2, ax2 = plt.subplots(nrows=len(tasks), figsize=figsize2)
        ax2[0].scatter(D_given_T_probs, T_probs, s=5, alpha=alpha)
        ax2[1].scatter(D_and_T_probs, T_probs, s=5, alpha=alpha)
        ax2[2].scatter(D_and_T_probs, D_given_T_probs, s=5, alpha=alpha)
        ax2[0].set_title(f"{m}")  
        ax2[0].set_xlabel('p(Y=1|T=1,X)')
        ax2[1].set_xlabel('p(Y=1,T=1,X)')
        ax2[2].set_xlabel('p(Y=1,T=1|X)')
        ax2[0].set_ylabel('p(T=1|X)')  
        ax2[1].set_ylabel('p(T=1|X)')
        ax2[2].set_ylabel('p(Y=1|T=1,X)')
        fig2.tight_layout()
        fig2.subplots_adjust(top=top_adjust)
        plt.show()
        plt.close()
    return dict_df_labels, dict_df_probs, dict_models, df_pp

def getGroundTruth(u, p_T, p_D_given_T):
    return p_D_given_T-u*(1-p_T)

def getURange(D_given_T_m, alpha=None, X=None):
    epsilon=1e-10 # need epsilon for when the min of D_given_T is 0
    if alpha is None:
        max_val_u = np.min(D_given_T_m)
        # max_val_u = 1.0
        diff = max(max_val_u*DIFF_UNOBSERVABLES_PLOT,epsilon) # plot 10 steps
        return np.arange(0, max_val_u+diff, diff)[:,None], diff
    elif alpha[-1]=='sigmoid':
        assert X is not None, f"X is not passed for sigmoid alpha"
        # implement sigmoid- randomly sample beta, sort it and get sigmoid(beta*X)
        beta_prior=np.arange(0,1.2,0.1)
        beta = [np.random.uniform(beta_prior[i], beta_prior[i+1], size=X.shape[1]) for i in range(len(beta_prior)-1)]
        u_x_ls = [simulateAlpha(beta[i], X) for i in range(len(beta))]
        # plt.scatter(beta_prior[:-1], u_x_ls, alpha=0.1)
        # plt.show()
        u_x = np.stack((u_x_ls))
        diff = DIFF_UNOBSERVABLES_PLOT
        return u_x, diff
    else:
        u_x=np.stack(([a*D_given_T_m for a in alpha ]))
        return u_x, DIFF_UNOBSERVABLES_PLOT

def plotCorr_w_Unobs(dict_df_probs, models, title, tasks, corr_method=stats.pearsonr, alpha=None, figsize=(10,10),
                    top_adjust=0.9, loc="lower left", title_en=True, X=None, legend_ncol=1, custom_ticker=False, second_stage=False, **kwargs):
    """
    alpha = np.arange(0,1.1,0.1)
    """
    calibrated_p_T=kwargs.get('calibrated_p_T')
    calibrated_p_D_T1=kwargs.get('calibrated_p_D_T1')
    savefig_path=kwargs.get('savefig_path')
    fig= plt.figure(figsize=figsize)
    assert corr_method in [stats.pearsonr, stats.spearmanr]
    if calibrated_p_T is not None: second_stage=True
    if corr_method==stats.pearsonr:
        title_extended=' '.join(['Pearson', '_second_stage_', str(second_stage), title])
    elif corr_method==stats.spearmanr:
        title_extended=' '.join(['Spearman', '_second_stage_', str(second_stage), title])

    assert 'T' in tasks # These two must be at least in tasks
    assert 'D|T' in tasks
    probs_task_model={}
    for i, m in enumerate(models):
        corr_ls, probs_task_model[m] = {}, {}
        for t in tasks:
            if t!='product_T_D_given_T':
                probs_task_model[m][t] = np.array(list(dict_df_probs[m][t].values))
            corr_ls[t]=[]
                
        if calibrated_p_D_T1 is not None:
            uRange, diff=getURange(calibrated_p_D_T1, alpha, X=X)
        else:
            uRange, diff=getURange(probs_task_model[m]['D|T'], alpha, X=X)
        ax = plt.subplot(len(models), 1, i+1)
        # ax.set_xlabel(r'$u$', fontsize=25)
        if alpha is not None and X is None:
            assert uRange.shape[0]==len(alpha)
            assert uRange.shape[1]==len(probs_task_model[m]['D|T'])
        ax.set_xlabel(r'$\alpha$', fontsize=25)
        groundTruthProbs_ls=[]
        for k in range(uRange.shape[0]):
            if calibrated_p_T is not None and calibrated_p_D_T1 is not None:
                groundTruthProbs=getGroundTruth(uRange[k,:], calibrated_p_T, calibrated_p_D_T1)
            else:
                groundTruthProbs=getGroundTruth(uRange[k,:], probs_task_model[m]['T'], probs_task_model[m]['D|T'])
            groundTruthProbs_ls.append(groundTruthProbs)
        for t in tasks:
            for k in range(uRange.shape[0]):
                if t=='product_T_D_given_T':
                    corr_task = partial(corr_method, probs_task_model[m]['T']*probs_task_model[m]['D|T'])
                else:
                    corr_task = partial(corr_method, probs_task_model[m][t])
                
                corr_ls[t].append(corr_task(groundTruthProbs_ls[k])[0])

            if alpha is not None:
                ax.plot(np.arange(0,1.1,0.1), corr_ls[t], marker='.',markersize=12, label=LABEL_DICT_SECOND_STAGE[t])
            else:
                ax.plot(uRange, corr_ls[t], marker='.',markersize=12, label=LABEL_DICT_SECOND_STAGE[t])
        # print the point of cross-over
        crossover_pt = np.argwhere(np.array(corr_ls['T'])==np.array(corr_ls['D|T']))
        if len(crossover_pt)>0: 
            print(f"the value of crossover is at alpha={uRange[crossover_pt,:]}")
        else:
            print(f"lines did not cross")
        # ax[i].set_ylabel(f"Correlation for model:{m}")
        if title_en: ax.set_ylabel(title, fontsize=20)
        if calibrated_p_T is None:
            for k in range(uRange.shape[0]):
                if alpha is None: 
                    corr_T = partial(corr_method, probs_task_model[m]['T'])
                    corr_T_D_given_T = corr_T(probs_task_model[m]['D|T'])[0]
                    
                    numerator = (np.std(probs_task_model[m]['D|T']))+(uRange[k,:]*corr_T_D_given_T*np.std(probs_task_model[m]['T']))
                    denominator = (np.std(probs_task_model[m]['D|T'])*corr_T_D_given_T)+(uRange[k,:]*np.std(probs_task_model[m]['T']))
                    ratio= numerator/denominator
                    if corr_method==stats.pearsonr:
                        assert math.isclose((corr_ls['D|T'][k])/(corr_ls['T'][k]), ratio)
                    if uRange[k,:]==0:
                        np.testing.assert_array_equal(groundTruthProbs_ls[k], probs_task_model[m]['D|T'])
                    if uRange[k,:]==1:
                        np.testing.assert_array_equal(groundTruthProbs_ls[k], (probs_task_model[m]['D|T']-(1-probs_task_model[m]['T'])))
                else:
                    if np.array_equal(uRange[k,:], np.zeros_like(uRange[k,:])):
                        np.testing.assert_array_equal(groundTruthProbs_ls[k], probs_task_model[m]['D|T'], err_msg=f"{k}, {probs_task_model[m]['D|T'].shape}, {uRange[k,:].shape}")
                    if np.array_equal(uRange[k,:], probs_task_model[m]['D|T']): # corresponds to alpha=1
                        np.testing.assert_allclose(groundTruthProbs_ls[k], (probs_task_model[m]['D|T']*(probs_task_model[m]['T'])))        
        #plot the line if alpha is None, where u = std of p(D|T)/std of p(T)
        if alpha is None:
            # put the asserts for u=0,1 and also when alpha is not None, for alpha=0,1 
            rank_D= stats.rankdata(probs_task_model[m]['D|T'])
            rank_T = stats.rankdata(probs_task_model[m]['T'])
            cov_D=np.cov(rank_D)
            cov_T=np.cov(rank_T)
            print(f"{np.cov(rank_D,rank_T)/(np.sqrt(cov_D)*np.sqrt(cov_T))}, spearman from lib:{stats.spearmanr(probs_task_model[m]['D|T'], probs_task_model[m]['T'])}, var of D:{cov_D}, var of T :{cov_T}")
            print(f" the point of crossover would be :{np.std(probs_task_model[m]['D|T'])/np.std(probs_task_model[m]['T'])} with std for p_D_T1:{np.std(probs_task_model[m]['D|T'])} and std for p_T:{np.std(probs_task_model[m]['T'])}")
            if (np.std(probs_task_model[m]['D|T'])/np.std(probs_task_model[m]['T']))<=uRange[-1]:
                ax.axvline(np.std(probs_task_model[m]['D|T'])/np.std(probs_task_model[m]['T']), ymin=0.0, ymax=1.0 , linestyle='--', color='black', label=r'$u = \sigma_{p_D}/\sigma_{p_T}$') 
            ax.set_xlim(0-diff, uRange[-1]+diff)
            if custom_ticker: ax.set_xticks(ax.get_xticks()[1::2])
        else:
            ax.set_xlim(0-diff, 1+diff)        
        ax.set_ylim(0-diff,1+diff)
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=15)
        if (alpha is not None) and (alpha[-1]=='sigmoid'):
            ax.set_xlabel(r'$\beta$', fontsize=25)
    
    if title_en: ax.legend(loc=loc, fontsize=19, ncol=legend_ncol)
    fig.tight_layout()
    fig.subplots_adjust(top=top_adjust)
    if savefig_path is not None:
        plt.savefig(f"{osp.join(savefig_path, ''.join([str(title_extended),'.pdf']))}")
    plt.show()
    plt.close()

def plotDistributionProbs(dict_df_probs, models, title, tasks, figsize=(10,10), top_adjust=0.9):
    fig = plt.figure(figsize=figsize)
    for i, m in enumerate(models):
        probs_tasks = {}
        for t in tasks:
            probs_tasks[t] = np.array(list(dict_df_probs[m][t].values))
        ax=plt.subplot(len(models),1,i+1)
        ax.violinplot([probs_tasks[t] for t in tasks] , 
                        showmeans=False, showmedians=True,showextrema=True)
        ax.set_xticks(np.arange(1, len(tasks) + 1), labels=[LABEL_DICT_SECOND_STAGE[t] for t in tasks], fontsize=15, rotation=45)
        ax.set_ylabel(title, fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=top_adjust)
    # plt.suptitle(title)
    plt.show()
    plt.close()

def get_ohe(df, cat_feats):
    for f in cat_feats:
        ohe_df=pd.get_dummies(df[f], prefix=f, drop_first=True).astype('bool')
        df=df.drop(f, axis=1)
        df=df.join(ohe_df)
        df.reset_index(drop=True, inplace=True)
    return df

def getClippedProbs(model, train_X, train_y_T, epsilon=0.05, cal_X=None, cal_y_T=None, **kwargs):
    """
    Clip the probabilities, since inverse of them will be used for IPW
    and this avoids blow up of weights
    """
    cv_method=kwargs.get('prefit')
    train_probs= model.predict_proba(train_X)[:,1]
    fig1, ax1 = plt.subplots(nrows=2, figsize=(10,10))
    if cal_X is not None:
        assert cal_y_T is not None, f"need to pass cal_y_T to be able to calibrate"
        if cv_method is not None:
            print(f"calibrating by cv method = prefit")
            raise ValueError("Not implemented")
        else:
            print(f"calibrating by bins using cal data")
            cal_probs=model.predict_proba(cal_X)[:,1]
            train_probs = calibrate_by_bin(cal_probs, cal_y_T, train_probs.copy(), train_y_T.copy(), n_bins=20)
    plotCalibrationPlots(train_probs, train_y_T, None, 'T', ax1[0], ax1[1], n_bins=20)
    fig1.show()
    # now subset for T=1 only
    train_y_ipw = train_y_T[train_y_T==1]
    train_X_ipw = train_X[train_y_T==1]
    probs_ipw = train_probs[train_y_T==1]
    probs_ipw=np.clip(probs_ipw, a_min=epsilon, a_max=1-epsilon)
    fig2, ax2 = plt.subplots()
    ax2.hist(probs_ipw)
    return probs_ipw, train_X_ipw

def getClippedProbsMedical(model, train_X, train_y_T, epsilon=0.05, cal_X=None, cal_y_T=None, **kwargs):
    """
    Clip the probabilities, since inverse of them will be used for IPW
    and this avoids blow up of weights
    """
    cv_method=kwargs.get('prefit')
    train_probs= model.predict_proba(train_X)[:,1]
    fig1, ax1 = plt.subplots(nrows=2, figsize=(10,10))
    if cal_X is not None:
        assert cal_y_T is not None, f"need to pass cal_y_T to be able to calibrate"
        if cv_method is not None:
            print(f"calibrating by cv method = prefit")
            model= CalibratedClassifierCV(model, cv='prefit', method="sigmoid") 
            model.fit(cal_X, cal_y_T)
            train_probs = model.predict_proba(train_X)[:,1]
    plotCalibrationPlots(train_probs, train_y_T, None, 'T', ax1[0], ax1[1], n_bins=20)
    fig1.show()
    # now subset for T=1 only
    
    train_X_ipw = train_X[train_y_T==1]
    probs_ipw = train_probs[train_y_T==1]
    probs_ipw=np.clip(probs_ipw, a_min=epsilon, a_max=1-epsilon)
    fig2, ax2 = plt.subplots()
    ax2.hist(probs_ipw)
    return probs_ipw, train_X_ipw

def getResiduals(prob_D_and_T, prob_D_given_T, prob_T):
    """
    Find and plot the residual of estimated p(A,B) and multiplication of estimators of p(A|B).P(B)
    """
    residuals=prob_D_and_T-(prob_D_given_T*prob_T)
    # sns.displot(residuals, kind='kde')
    plt.hist(residuals, bins=10)
    plt.xlabel("estimated p(D,T|X) - (estimated p(D|T,X)* estimated p(T|X))")
    plt.show()
    plt.close()
    # plt.violinplot(residuals,showmeans=False, showmedians=True,showextrema=True)
    # plt.xlabel("distribution of estimated p(D,T|X) - (estimated p(D|T,X)* estimated p(T|X))")
    # plt.show()  
    # plt.close()

#util funcs
def saveFile(path, file, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    if '.npy' in filename:
        with open(''.join([path,filename]), 'wb') as f:
            np.save(f, file)
    elif '.pkl' in filename:
        with open(osp.join(path, filename), 'wb') as f:
            pickle.dump(file, f)
    elif '.csv' in filename:
        file.to_csv(path, index=False)

def loadFile(path, filename):
    assert os.path.exists(path), f" {path} does not exist"
    if '.npy' in filename:
        with open(''.join([path,filename]), 'rb') as f:
            file = np.load(f, allow_pickle=True)
    elif '.pkl' in filename:
        with open(osp.join(path, filename), 'rb') as f:
            file = pickle.load(f)
    elif '.csv' in filename:
        file= pd.read_csv(osp.join(path, filename))
    return file

def getTrainTestIdx(idxs=None, df=None, train_size=0.5):
    if idxs is None:
        assert df is not None
        idxs = np.array(df.index)
    assert idxs is not None, f" pass either the dataframe or indexs directly"
    train_idxs = np.random.choice(idxs, size=int(len(idxs)*train_size), replace=False)
    test_idxs = np.setdiff1d(idxs, train_idxs)
    assert len(np.intersect1d(train_idxs, test_idxs))==0
    return train_idxs, test_idxs

# Longer outcomes or equity impact
def StackedBarDI(dict_df_probs, df, models, tasks, title, figsize=(15,7), top_adjust=0.9):
    """
    For stop and frisk dataset, find the p(race|top k% weapons found)
    """
    fig= plt.figure(figsize=figsize)
    race_lbl = ['Black', 'Hispanic', 'White'] 
    # k_range= np.arange(0.5,1.0,0.1)
    k_range=[0.75]
    color_tab10_v1 = plt.cm.tab10.colors
    
    colors_map={'Black':color_tab10_v1[1], 'Hispanic':color_tab10_v1[0], 'White':color_tab10_v1[2]}
    width = 0.3
    fracs={}
    for i, m in enumerate(models):
        df_probs = dict_df_probs[m].copy()
        for j, t in enumerate(tasks):
            ax = plt.subplot(len(models), len(tasks), i*len(tasks)+(j+1))
            probs=df_probs[t].values
            sorted_idxs = np.argsort(probs)[::-1]
            fracs[t]={'Black':[], 'Hispanic':[], 'White':[]}
            bottom={'Black':[], 'Hispanic':[], 'White':[]}
            bottom['Black']=None
            for a,p in enumerate(k_range):
                top_k_idxs = sorted_idxs[:int((1-p)*len(probs))]
                
                for k,r in enumerate(race_lbl):
                    df_top_k_probs_race = df.iloc[top_k_idxs].loc[df["suspect.race"]==r]
                    fracs[t][r].append(len(df_top_k_probs_race)/len(top_k_idxs))
                    if k==2:
                        bottom['White']=[sum(x) for x in zip(fracs[t][race_lbl[0]], fracs[t][race_lbl[1]])]
                    elif k==1:
                        bottom['Hispanic']=fracs[t][race_lbl[0]]
                print(f" top {(1-p)*100:.2f} % samples mean :{probs[top_k_idxs].mean()}")
            for r in race_lbl:
                ax.bar(["".join([str(f"{(1-p)*100:.2f}"), '%']) for p in k_range], 
                        fracs[t][r], bottom=bottom[r], label=r, width=0.6, align='center', color=colors_map[r] )   
            
            ax.set_title(LABEL_DICT_SECOND_STAGE[t])
            ax.tick_params(axis='y', labelsize=15)
        # ax.set_ylabel(m, fontsize=15)
            
    fig.tight_layout()
    fig.subplots_adjust(top=top_adjust)
    plt.suptitle(title, fontsize=15)
    plt.legend(fontsize=15, ncol=3, bbox_to_anchor=(0, -0.2), loc='center')
    plt.show()
    plt.close()
    return fracs

def simulateAlpha(beta, X):
    """
    alpha = 1/(1+exp(-X@beta))
    """
    reg = X@beta
    assert reg.shape==(X.shape[0],)
    p = 1/(1+np.exp(-reg.reshape(-1,)))
    assert np.all(p>=0) and np.all(p<=1)
    assert p.shape==reg.shape
    return p

def splitFunc(df, groupName, test_size=0.5):
    df[groupName]=df[groupName].apply(lambda x: "UNK" if x!=x else x)
    df.reset_index(inplace=True)
    df_grouped=df[['index', groupName]].groupby([groupName], dropna=False).agg(['count']).reset_index()
    df_grouped.columns=[groupName, 'prop']
    df_grouped['prop']=df_grouped['prop'].apply(lambda x:x/len(df))
    df_grouped = df_grouped.sample(frac=1).reset_index(drop=True)
    cumProp=0
    i=0
    train_groups=[]
    amt_later=0
    if len(df.loc[df[groupName]=="UNK"])>0:
        amt_later = df_grouped.loc[df_grouped[groupName]=="UNK"]['prop'].item()
    while (cumProp<=(1-test_size-(amt_later*0.5))):
        if df_grouped.iloc[i][groupName]=="UNK":
            i+=1
            continue
        cumProp+=df_grouped.iloc[i]['prop']
        train_groups.append(df_grouped.iloc[i][groupName])
        i+=1
    test_groups = set(df_grouped[groupName])-set(train_groups)-set(["UNK"])
    assert len(np.intersect1d(np.array(train_groups), np.array(list(test_groups))))==0
    print(cumProp, i)
    train_idxs=np.array([])
    if len(df.loc[df[groupName]=="UNK"])>0:
        train_idxs = np.array(df.loc[df[groupName]=="UNK"].sample(frac=0.5).index)
    train_idxs=np.concatenate((train_idxs, np.array(df.loc[df[groupName].isin(train_groups)].index)))
    train_idxs = train_idxs.astype(int)
    idxs=np.arange(len(df))
    test_idxs=np.setdiff1d(idxs, train_idxs)
    print(len(train_idxs), len(test_idxs), len(idxs))
    assert len(np.intersect1d(train_idxs, test_idxs))==0
    return train_idxs, test_idxs

def secondStageDataGen(test_idxs, p_T, p_D_T1, train_size, dataset="other", df=None, **kwargs):
    """
    Given test set, divide into train and test again for second stage
    Based on initial probability estimates from the first stage model, generate
    T ~ Bernoulli(p(T=1|X))
    D_T1 ~ Bernoulli(p(D=1|T=1,X))
    Args: 
        test_idxs : first stage test idxs that are absolute to the full dataset (used only for their length to get 
        the train second stage and test second stage split)
        train_prop: [0,1] proportion to train for second stage
        p_T: calibrated p(T=1|X) probabilities for the test set from first stage
        p_D_T1: calibrated p(D=1|T=1,X) probabilities for the test set from first stage
    Returns:
        train_second_stage_idxs, test_second_stage_idxs -  relative to the test set, not absolute
        train_second_stage_T, test_second_stage_T
        train_second_stage_D_T1, test_second_stage_D_T1
        train_second_stage_D_and_T, test_second_stage_D_and_T
    """
    cal_size=kwargs.get('cal_size')
    train_second_stage, test_second_stage, cal_second_stage = {}, {}, {}
    if str(dataset).lower()=='inspections':
        assert df is not None
        print("For Inspections dataset")
        train_second_stage['idxs'], test_second_stage['idxs'] = splitFunc(df.iloc[test_idxs], groupName='census_tract', test_size=1-train_size)
    else:
        train_second_stage['idxs'], test_second_stage['idxs'] = getTrainTestIdx(idxs=np.arange(len(test_idxs)), train_size=train_size)
    if cal_size is not None:
        if str(dataset).lower()=='inspections':
            assert df is not None
            print("For Inspections dataset")
            cal_second_stage['idxs'], test_second_stage['idxs'] = splitFunc(df.iloc[test_second_stage['idxs']], groupName='census_tract', test_size=1-cal_size)
        else:
            cal_second_stage['idxs'], test_second_stage['idxs'] = getTrainTestIdx(idxs=test_second_stage['idxs'].copy(), train_size=cal_size)
        assert len(np.intersect1d(cal_second_stage['idxs'], test_second_stage['idxs']))==0
        assert len(np.intersect1d(cal_second_stage['idxs'], train_second_stage['idxs']))==0
    assert len(np.intersect1d(train_second_stage['idxs'], test_second_stage['idxs']))==0
    

    train_second_stage['T'] = np.random.binomial(1, p_T[train_second_stage['idxs']], len(train_second_stage['idxs'])).astype(dtype=int)
    test_second_stage['T'] = np.random.binomial(1, p_T[test_second_stage['idxs']], len(test_second_stage['idxs'])).astype(dtype=int)
    if cal_size is not None:
        cal_second_stage['T'] = np.random.binomial(1, p_T[cal_second_stage['idxs']], len(cal_second_stage['idxs'])).astype(dtype=int)

    train_second_stage['D'] = np.random.binomial(1, p_D_T1[train_second_stage['idxs']], len(train_second_stage['idxs'])).astype(dtype=float)
    test_second_stage['D'] = np.random.binomial(1, p_D_T1[test_second_stage['idxs']], len(test_second_stage['idxs'])).astype(dtype=float)
    if cal_size is not None:
        cal_second_stage['D'] = np.random.binomial(1, p_D_T1[cal_second_stage['idxs']], len(cal_second_stage['idxs'])).astype(dtype=float)

    np.testing.assert_array_equal (np.unique(train_second_stage['T']), np.array([0, 1])), f"{np.unique(train_second_stage['T'])}"
    np.testing.assert_array_equal (np.unique(train_second_stage['D']), np.array([0, 1])), f"{np.unique(train_second_stage['D'])}"
    assert len(train_second_stage['T'][train_second_stage['T']==0])>0, f"no T=0 samples exist in train set"
    train_second_stage['D'][train_second_stage['T']==0]=np.nan
    
    assert (np.isnan(train_second_stage['D'])).sum()==len(train_second_stage['T'][train_second_stage['T']==0]), f"{(np.isnan(train_second_stage['D'])).sum()}, {len(train_second_stage['T'][train_second_stage['T']==0])}"
    train_second_stage['D_and_T'] = (train_second_stage['T']==1) & (train_second_stage['D']==1)
    assert train_second_stage['D_and_T'].sum()==(train_second_stage['D'][~np.isnan(train_second_stage['D'])]).sum(), f" D_and_T sum does not match sum where D=1,T=1"
    train_second_stage['D_and_T'] = train_second_stage['D_and_T'].astype(int)
    np.testing.assert_array_equal (np.unique(train_second_stage['D_and_T']), np.array([0, 1])), f"{np.unique(train_second_stage['D_and_T'])}"

    np.testing.assert_array_equal (np.unique(test_second_stage['T']), np.array([0, 1])), f"{np.unique(test_second_stage['T'])}"
    np.testing.assert_array_equal (np.unique(test_second_stage['D']), np.array([0, 1])), f"{np.unique(test_second_stage['D'])}"
    assert len(test_second_stage['T'][test_second_stage['T']==0])>0, f"no T=0 samples exist in test set"
    test_second_stage['D'][test_second_stage['T']==0]=np.nan

    assert (np.isnan(test_second_stage['D'])).sum()==len(test_second_stage['T'][test_second_stage['T']==0]), f"{(np.isnan(test_second_stage['D'])).sum()}, {len(test_second_stage['T'][test_second_stage['T']==0])}"
    test_second_stage['D_and_T'] = ((test_second_stage['T']==1) & (test_second_stage['D']==1)).astype(int)
    assert test_second_stage['D_and_T'].sum()==(test_second_stage['D'][~np.isnan(test_second_stage['D'])]).sum()
    np.testing.assert_array_equal (np.unique(test_second_stage['D_and_T']), np.array([0, 1])), f"{np.unique(test_second_stage['D_and_T'])}"

    if (cal_size is not None):
        np.testing.assert_array_equal (np.unique(cal_second_stage['T']), np.array([0, 1])), f"{np.unique(cal_second_stage['T'])}"
        np.testing.assert_array_equal (np.unique(cal_second_stage['D']), np.array([0, 1])), f"{np.unique(cal_second_stage['D'])}"
        assert len(cal_second_stage['T'][cal_second_stage['T']==0])>0, f"no T=0 samples exist in test set"
        cal_second_stage['D'][cal_second_stage['T']==0]=np.nan

        assert (np.isnan(cal_second_stage['D'])).sum()==len(cal_second_stage['T'][cal_second_stage['T']==0]), f"{(np.isnan(cal_second_stage['D'])).sum()}, {len(cal_second_stage['T'][cal_second_stage['T']==0])}"
        cal_second_stage['D_and_T'] = ((cal_second_stage['T']==1) & (cal_second_stage['D']==1)).astype(int)
        assert cal_second_stage['D_and_T'].sum()==(cal_second_stage['D'][~np.isnan(cal_second_stage['D'])]).sum()
        np.testing.assert_array_equal (np.unique(cal_second_stage['D_and_T']), np.array([0, 1])), f"{np.unique(cal_second_stage['D_and_T'])}"
    
    return train_second_stage, test_second_stage, cal_second_stage


def getLabels(taskLabels, taskPids, pids):
    """
    HiRID dataset has a custom pipeline. In order to have the same order for p(T) and p(D|T) probs,
    this function matches based on the patient ids which are unique identifier for a record.
    Eventhough the function is labeled getLabels it is used to match on indexes
    Args:
        taskLabels: labels for a specific task, eg: labels for p(D|T)
        taskPids: patient ids for the speciifc task
        pids: master patient ids. These are the list of all patient ids
    Returns:
        out_arr: labels for the specific task returned in the order of mater pids
    """
    if len(taskLabels.shape)>1:
        n_dim=taskLabels.shape[1]
        out_arr = np.zeros((len(pids),n_dim))
    else:
        out_arr = np.zeros((len(pids),))
    out_arr[:]=np.nan
    for i, p in enumerate(taskPids):
        idx_match=np.where(pids==p)[0]
        if len(idx_match)>0:
            out_arr[idx_match.item()]=taskLabels[i]
    return out_arr

def getCorrMedical(models, tasks, X, clf_dict, df, test_idxs, dict_df_labels, dict_df_probs, dict_models, df_pp, probs_path, corr_method=stats.pearsonr, 
            calibrate=False, figsize1=(15,35), figsize2=(10,10), top_adjust=0.85, alpha=1.0, calibrate_method="sigmoid", **kwargs):
    epsilon = kwargs.get('epsilon')
    test_second_stage = kwargs.get('test_second_stage')
    cal_second_stage = kwargs.get('cal_second_stage')
    cal_idxs = kwargs.get('cal_idxs')
    cal_X = kwargs.get('cal_X')

    for col, m in enumerate(models):
        dict_df_labels[m]= pd.DataFrame([])
        dict_df_probs[m]= pd.DataFrame([])
        dict_models[m]= {}
        fig1, ax1 = plt.subplots(nrows=2*len(tasks), figsize=figsize1)
        for row, t in enumerate(tasks):
            if test_second_stage is None:
                assert df is not None, f"need to pass dataframe for first stage"
                model , labels= loadModelData(m, t, df, test_idxs, clf_dict)
            else:
                model , labels= loadModelData_secondStage(m, t, test_second_stage, clf_dict)
            dict_models[m][str(t)]=model
            dict_df_labels[m][str(t)]=labels
            predict_probs = model.predict_proba(X)
            assert len(predict_probs)==len(X)
            assert predict_probs.shape[1]==2 # predicting binary class probbs, so index zero for class0, index 1 for class1
            probs=predict_probs[:,1]
            if (calibrate) and (cal_X is not None):
                print(f" calibrating by prefit data with cal data and calibratedClassifierCV")
                model= CalibratedClassifierCV(model, cv='prefit', method=calibrate_method) 
                if test_second_stage is None:
                    assert df is not None, f"need to pass dataframe for first stage"
                    assert cal_idxs is not None, f" need to pass cal_idxs to extract labels from df"
                    _, val_labels=loadModelData(m, t, df, cal_idxs, clf_dict)
                else:
                    assert cal_second_stage is not None, f"Need to pass econd stage labels for cal data"
                    _, val_labels=loadModelData_secondStage(m, t, cal_second_stage, clf_dict)
                #drop nan values from calibration set
                val_labels_not_nan=val_labels[~np.isnan(val_labels)]
                cal_X_not_nan=cal_X[~np.isnan(val_labels)]
                model.fit(cal_X_not_nan, val_labels_not_nan) # fit on X_val_feat
                val_probs = model.predict_proba(cal_X)[:,1]
                assert len(val_probs)==len(cal_X)
                probs = model.predict_proba(X)[:,1]
            dict_df_probs[m][str(t)]=probs
            probs = probs[~np.isnan(labels)]
            labels = labels[~np.isnan(labels)]
            auc_score, pr_score, balanced_acc = getMetrics(labels, probs)[:3]
            print(f"classification report for model {m}, task {t}: \n {getMetrics(labels, probs)[3]}")
            df_pp=df_pp.append({'AUC' :auc_score ,'PR':pr_score,'BalancedAcc':balanced_acc, 'modelName':m,'rowName':t},ignore_index=True)
            
            assert len(dict_df_probs[m][str(t)])==len(X)
            
            plotCalibrationPlots(probs, labels, m, t, ax1[2*row], ax1[(2*row)+1])
            #save probs
            saveFile(osp.join(probs_path, f"probs_{t}"), dict_df_probs[m][t], f"/probs.npy")
        fig1.tight_layout()
        fig1.subplots_adjust(top=top_adjust)
        plt.show()
        plt.close()
        # get the correlation
        T_probs = dict_df_probs[m]['T']
        D_given_T_probs = dict_df_probs[m]['D|T']
        D_and_T_probs = dict_df_probs[m]['D_and_T']
        corr_T=partial(corr_method, T_probs)
        corr_D_and_T=partial(corr_method, D_and_T_probs)
        print(f"T and D|T probs for model {m}: {corr_T(D_given_T_probs)}")
        print(f"T and D,T probs for model {m}: {corr_T(D_and_T_probs)}")
        print(f"D,T and D|T probs for model {m}: {corr_D_and_T(D_given_T_probs)}")
        fig2, ax2 = plt.subplots(nrows=len(tasks), figsize=figsize2)
        ax2[0].scatter(D_given_T_probs, T_probs, s=5, alpha=alpha)
        ax2[1].scatter(D_and_T_probs, T_probs, s=5, alpha=alpha)
        ax2[2].scatter(D_and_T_probs, D_given_T_probs, s=5, alpha=alpha)
        ax2[0].set_title(f"{m}")  
        ax2[0].set_xlabel('p(Y=1|T=1,X)')
        ax2[1].set_xlabel('p(Y=1,T=1,X)')
        ax2[2].set_xlabel('p(Y=1,T=1|X)')
        ax2[0].set_ylabel('p(T=1|X)')  
        ax2[1].set_ylabel('p(T=1|X)')
        ax2[2].set_ylabel('p(Y=1|T=1,X)')
        fig2.tight_layout()
        fig2.subplots_adjust(top=top_adjust)
        plt.show()
        plt.close()
    return dict_df_labels, dict_df_probs, dict_models, df_pp

# long term outcomes

def set_axis_style(ax, labels, title, xlabel, ylabel):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(xlabel+" (Binary Label)")
    ax.set_title(title)
    ax.set_ylabel(ylabel)    

def getCommonPids(pid_ls1, pid_ls2):
    ret_pids1, ret_pids2=[],[]
    intersect_pids=np.intersect1d(pid_ls1, pid_ls2)
    for pid in intersect_pids:
        ret_pids1.append(np.argwhere(pid_ls1==pid).item())
        ret_pids2.append(np.argwhere(pid_ls2==pid).item())
    assert len(intersect_pids)==len(ret_pids1)==len(ret_pids2)
    assert len(intersect_pids)>1
    return intersect_pids, ret_pids1, ret_pids2

def getLongerOutcomesMetrics(modelNames, tasks, estNames, outcome_D, title, xLabel, yLabel, df_metrics,
 dict_df_ids_cutoff, dict_df_labels_cutoff, dict_df_probs_cutoff, dict_df_labels, dict_df_ids, mort_status_rel_stay, **kwargs):
    """ Get metrics between task A and B and longer time outcomes
    args:
        estimators[0][mName]={}, estimators[1][mName]={}
    """
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharey=True)
    i,j=0,0
        
    outcome_D_name=kwargs.get('outcome_D_name')
    ipw=kwargs.get('ipw')
    for mName in modelNames:
        
        idx_untested=np.where(dict_df_labels_cutoff['T']==0)[0]
        
        T_labels = np.array(dict_df_labels['T'].values.tolist())
        T_ids = np.array(dict_df_ids['T'].values.tolist())
        
        untested_T_cutoff_ids = np.array(dict_df_ids_cutoff['T'].values.tolist())
        
        T_labels = getLabels(T_labels, T_ids, untested_T_cutoff_ids)
        untested_T_labels=T_labels[idx_untested][~np.isnan(T_labels[idx_untested])]
        
        untested_T_probs = np.array(dict_df_probs_cutoff['T'].values.tolist())[idx_untested]
        untested_T_probs=untested_T_probs[~np.isnan(T_labels[idx_untested])]
        untested_D_probs = np.array(dict_df_probs_cutoff[str(tasks[1])].values.tolist())[idx_untested]
        untested_D_probs=untested_D_probs[~np.isnan(T_labels[idx_untested])] 
        
        assert len(untested_T_labels)==len(untested_T_probs)
        print(f"labels for untested T=0 pop for model {str(mName)}, :{len(untested_T_probs)}")
        print(f"correlation 1x1:"
        f"{stats.spearmanr(untested_T_labels, untested_T_probs)}")
        auc_1_1, pr_1_1, ba_1_1=getMetrics(untested_T_labels, untested_T_probs)[:3]
        cr_1_1=getMetrics(untested_T_labels, untested_T_probs)[-1]
        print(f"PR, balanced accuracy metrics for 1x1 :{str(mName)}, {pr_1_1},{ba_1_1} \n {cr_1_1}")
        print(f"auc score 1x1:{auc_1_1:.4f}")
        
        auc_2_1, pr_2_1, ba_2_1=getMetrics(untested_T_labels, untested_D_probs)[:3]
        cr_2_1=getMetrics(untested_T_labels, untested_D_probs)[-1]
        print(f"correlation 2x1:{stats.spearmanr(untested_T_labels, untested_D_probs)}")
        print(f"PR, balanced accuracy metrics for 2x1 :{str(mName)}, {pr_2_1},{ba_2_1} \n {cr_2_1}")
        print(f"auc score 2x1:{auc_2_1:.4f}")
        
        df_metrics=df_metrics.append({'AUC' :auc_1_1 ,'PR':pr_1_1,'BalancedAcc':ba_1_1, 'modelName':mName,'rowName':estNames[0],'colName':'T'},ignore_index=True)
        df_metrics=df_metrics.append({'AUC' :auc_2_1 ,'PR':pr_2_1,'BalancedAcc':ba_2_1,'modelName':mName,'rowName':estNames[1],'colName':'T'},ignore_index=True)

        ax[i,j].violinplot([untested_T_probs[untested_T_labels==i] for i in [0,1]], 
                           showmeans=False, showmedians=True,showextrema=True)
        ax[i+1,j].violinplot([untested_D_probs[untested_T_labels==i] for i in [0,1]],
                             showmeans=False, showmedians=True,showextrema=True)
        
        #2,2 and 1,2
        idx_untested_D_given_T = np.where((dict_df_labels_cutoff['T']==0) & 
                                            (T_labels==1)[0])
                
        T_ids = np.array(dict_df_ids['T'].values.tolist())
        D_given_T_labels = np.array(dict_df_labels[str(tasks[1])].values.tolist())
        D_given_T_labels = getLabels(D_given_T_labels, T_ids, untested_T_cutoff_ids)
        untested_D_given_T_labels = D_given_T_labels[idx_untested_D_given_T][~np.isnan(D_given_T_labels[idx_untested_D_given_T])]
        
        outcome_D_labels = np.array(dict_df_labels[outcome_D].values.tolist())
        outcome_D_labels = getLabels(outcome_D_labels, T_ids, untested_T_cutoff_ids)
        untested_outcome_D_labels=outcome_D_labels[idx_untested_D_given_T][~np.isnan(outcome_D_labels[idx_untested_D_given_T])]
        
        untested_T_probs = np.array(dict_df_probs_cutoff['T'].values.tolist())[idx_untested_D_given_T]
        untested_T_probs=untested_T_probs[~np.isnan(outcome_D_labels[idx_untested_D_given_T])]
        untested_D_probs = np.array(dict_df_probs_cutoff[str(tasks[1])].values.tolist())[idx_untested_D_given_T]
        untested_D_probs=untested_D_probs[~np.isnan(outcome_D_labels[idx_untested_D_given_T])]
        
        print(f"labels for untested T=0 after 1 hour and D=1 pop for model {str(mName)}, :{len(untested_D_given_T_labels)}")
        print(f"correlation 1x2:"
        f"{stats.spearmanr(untested_outcome_D_labels, untested_T_probs)}")
        
        print(f"correlation 2x2:"
        f"{stats.spearmanr(untested_outcome_D_labels, untested_D_probs)}")
        auc_1_2, pr_1_2, ba_1_2=getMetrics(untested_outcome_D_labels, untested_T_probs)[:3]
        cr_1_2=getMetrics(untested_outcome_D_labels, untested_T_probs)[-1]
        print(f"auc score 1x2:{auc_1_2:.4f}")
        
        auc_2_2, pr_2_2, ba_2_2=getMetrics(untested_outcome_D_labels, untested_D_probs)[:3]
        cr_2_2=getMetrics(untested_outcome_D_labels, untested_D_probs)[-1]
        print(f"auc score 2x2:{auc_2_2:.4f}")
        print(f"PR, balanced accuracy metrics for 1x2 :{str(mName)}, {pr_1_2},{ba_1_2} \n {cr_1_2}")

        print(f"PR, balanced accuracy metrics for 2x2 :{str(mName)}, {pr_2_2},{ba_2_2} \n {cr_2_2}")

        df_metrics=df_metrics.append({'AUC' :auc_1_2 ,'PR':pr_1_2,'BalancedAcc':ba_1_2,'modelName':mName,'rowName':estNames[0],'colName':f"{outcome_D_name}"},ignore_index=True)
        df_metrics=df_metrics.append({'AUC' :auc_2_2 ,'PR':pr_2_2,'BalancedAcc':ba_2_2,'modelName':mName,'rowName':estNames[1],'colName':f"{outcome_D_name}"},ignore_index=True)

        ax[i,j+1].violinplot([untested_T_probs[untested_outcome_D_labels==i] for i in [0,1]], 
                             showmeans=False, showmedians=True, showextrema=True)
        ax[i+1,j+1].violinplot([untested_D_probs[untested_outcome_D_labels==i] for i in [0,1]],
                               showmeans=False, showmedians=True, showextrema=True)

        #1,3 and 2,3

        common_mort_pids, idx_labels_mort, idx_probs_mort =\
                getCommonPids(mort_status_rel_stay.loc[(~pd.isnull(mort_status_rel_stay['discharge_status']))\
                            & (mort_status_rel_stay['rel_datetime']>12)]['patientid'].values.tolist(),
                untested_T_cutoff_ids[idx_untested])
        mort_labels=np.array(mort_status_rel_stay.loc[(~pd.isnull(mort_status_rel_stay['discharge_status']))\
                            & (mort_status_rel_stay['rel_datetime']>12)]['status'].values.tolist())[idx_labels_mort]

        assert np.isnan(mort_labels).sum()==0
        
        untested_T_probs = np.array(dict_df_probs_cutoff['T'].values.tolist())[idx_untested]
        untested_D_probs = np.array(dict_df_probs_cutoff[str(tasks[1])].values.tolist())[idx_untested]

        print(f"labels for untested T=0 pop for model {str(mName)} and mort status, :{len(common_mort_pids)}")

        print(f"correlation 1x3:"
        f"{stats.spearmanr(mort_labels, untested_T_probs[idx_probs_mort])}")
        auc_1_3, pr_1_3, ba_1_3=getMetrics(mort_labels, untested_T_probs[idx_probs_mort])[:3]
        cr_1_3=getMetrics(mort_labels, untested_T_probs[idx_probs_mort])[-1]
        print(f"PR, balanced accuracy metrics for 1x3 :{str(mName)}, {pr_1_3},{ba_1_3} \n {cr_1_3}")
        print(f"correlation 2x3:"
        f"{stats.spearmanr(mort_labels, untested_D_probs[idx_probs_mort])}")
        auc_2_3, pr_2_3, ba_2_3=getMetrics(mort_labels, untested_D_probs[idx_probs_mort])[:3]
        cr_2_3=getMetrics(mort_labels, untested_D_probs[idx_probs_mort])[-1]
        print(f"PR, balanced accuracy metrics for 2x3 :{str(mName)}, {pr_2_3},{ba_2_3} \n {cr_2_3}")
        print(f"auc score 1x3:{auc_1_3:.4f}")
        print(f"auc score 2x3:{auc_2_3:.4f}")
        df_metrics=df_metrics.append({'AUC' :auc_1_3 ,'PR':pr_1_3,'BalancedAcc':ba_1_3,'modelName':mName,'rowName':estNames[0],'colName':'Mortality'},ignore_index=True)
        df_metrics=df_metrics.append({'AUC' :auc_2_3 ,'PR':pr_2_3,'BalancedAcc':ba_2_3,'modelName':mName,'rowName':estNames[1],'colName':'Mortality'},ignore_index=True)

        ax[i,j+2].violinplot([untested_T_probs[idx_probs_mort][mort_labels==i] for i in [0,1]], 
                             showmeans=False, showmedians=True, showextrema=True)
        ax[i+1,j+2].violinplot([untested_D_probs[idx_probs_mort][mort_labels==i] for i in [0,1]],
                               showmeans=False, showmedians=True, showextrema=True)
        print(f"len of alive:{len([untested_T_probs[idx_probs_mort][mort_labels==i] for i in [0]][0])}")
        print(f"len of dead:{len([untested_T_probs[idx_probs_mort][mort_labels==i] for i in [1]][0])}")
        # set style for the axes
        labels = ['0', '1']
        
        for k,a in enumerate([ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2]]):
            set_axis_style(a, labels, title[k], xLabel[k], yLabel[k])
        fig.tight_layout()
        plt.show()
    plt.close()
    return df_metrics