import seaborn
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
seaborn.set_context('paper', font_scale=3.8)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.weight'] = 'bold'


def plot_complete_information(dataset):
    ret = []
    for Lambda in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2, 4, 6, 8, 10]:
        data = pd.read_csv('../result/redwine_Lambda=%.2f.csv' % Lambda).iloc[:, 1:]
        data['Lambda'] = pd.Series(Lambda, index=data.index)
        ret.append(data)
    ret = pd.concat(ret)
    ret = ret.iloc[:, :-1]

    ret.set_value(ret[ret['type'].isin(['game'])].index, 'type', 'MLSG')
    ret.set_value(ret[ret['type'].isin(['normal'])].index, 'type', 'OLS')
    ret.set_value(ret[ret['type'].isin(['lasso'])].index, 'type', 'Lasso')
    ret.set_value(ret[ret['type'].isin(['ridge'])].index, 'type', 'Ridge')

    col = ['Lasso', 'OLS', 'Ridge', 'MLSG']
    error = ret.groupby('type').sem().transpose()[col] * 1.96
    ret.groupby('type').mean().transpose()[col].plot(kind='bar', colormap='Accent', yerr=error, width=0.7)
    plt.legend(title=False)
    plt.xlabel('$\\beta$')
    plt.ylabel('RMSE')
    plt.xticks(rotation='horizontal')
    plt.grid(linestyle='dotted')
    plt.tight_layout()
    plt.savefig('../result/%s_defender=attacker.pdf' % dataset)
    plt.show()


def plot_incomplete_information(dataset, estimateZ):
    ### defender'Lambda=0.5, defender's beta=0.8
    ### attacker's Lambda across a range. 

    ret = []
    for Lambda in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2, 4, 6, 8, 10]:
        data = pd.read_csv('../result/redwine_defenderLambda=0.50_defenderBeta=0.80_%s_attackerLambda=%.2f.csv' % (estimateZ, Lambda)).iloc[:, 1:]
        data['Lambda'] = pd.Series(Lambda, index=data.index)
        ret.append(data)
    ret = pd.concat(ret)

    MLSG = ret[ret['type'].isin(['MLSG'])].groupby('Lambda').mean()
    OLS = ret[ret['type'].isin(['OLS'])].groupby('Lambda').mean()
    Lasso = ret[ret['type'].isin(['Lasso'])].groupby('Lambda').mean()
    Ridge = ret[ret['type'].isin(['Ridge'])].groupby('Lambda').mean()

    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    yticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2, 4, 6, 8, 10]
    min_value = np.min((np.min(MLSG), np.min(OLS)))
    max_value = np.max((np.min(MLSG), np.max(OLS)))

    ax = seaborn.heatmap(MLSG, cmap="OrRd", xticklabels=xticks, yticklabels=yticks, vmin=min_value, vmax=max_value)
    ax.set_xlabel(r'$\mathbf{\beta}$', fontsize=50)
    ax.set_ylabel(r'$\mathbf{\lambda}$', rotation=45, fontsize=50)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.yticks(rotation=45)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig('../result/%s_%s_MLSG.pdf' % (dataset, estimateZ))
    plt.show()


    ax = seaborn.heatmap(OLS, cmap="OrRd", xticklabels=xticks, yticklabels=yticks, vmin=min_value, vmax=max_value)
    ax.set_xlabel(r'$\mathbf{\beta}$', fontsize=50)
    ax.set_ylabel(r'$\mathbf{\lambda}$', rotation=45, fontsize=50)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.yticks(rotation=45)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig('../result/%s_%s_OLS.pdf' % (dataset, estimateZ))
    plt.show()



    ax = seaborn.heatmap(Lasso, cmap="OrRd", xticklabels=xticks, yticklabels=yticks, vmin=min_value, vmax=max_value)
    ax.set_xlabel(r'$\mathbf{\beta}$', fontsize=50)
    ax.set_ylabel(r'$\mathbf{\lambda}$', rotation=45, fontsize=50)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.yticks(rotation=45)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig('../result/%s_%s_Lasso.pdf' % (dataset, estimateZ))
    plt.show()

    ax = seaborn.heatmap(Ridge, cmap="OrRd", xticklabels=xticks, yticklabels=yticks, vmin=min_value, vmax=max_value)
    ax.set_xlabel(r'$\mathbf{\beta}$', fontsize=50)
    ax.set_ylabel(r'$\mathbf{\lambda}$', rotation=45, fontsize=50)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.yticks(rotation=45)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig('../result/%s_%s_Ridge.pdf' % (dataset, estimateZ))
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    dataset = args.dataset

    plot_complete_information(dataset)
    plot_incomplete_information(dataset, 'overEstimateTarget')
    plot_incomplete_information(dataset, 'underEstimateTarget')



