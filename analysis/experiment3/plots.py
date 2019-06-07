import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SOA = {
    'ENZYMES': .65,
    'MUTAG': .889,
    'NCI1': .827,
    'NCI109': .824,
    'PROTEINS': .764,
    'PTC': .69
}

exps = {
    1: {
        'data': 'result1.csv',
        'title': '$pI + qA + (1 - q) D^{-\\frac{1}{2}} A D^{-\\frac{1}{2}}$\n(LR Decay)',
    },
    2: {
        'data': 'result2.csv',
        'title': '$(p I + (1 - p) D)^{-\\frac{1}{2}} (A + qI) (p I + (1 - p) D)^{-\\frac{1}{2}}$\n(LR Decay)',

    },
    3: {
        'data': 'result3.csv',
        'title': '$(p I + (1 - p) D)^{-\\frac{1}{2}} (A + qI) (p I + (1 - p) D)^{-\\frac{1}{2}}$\n(No LR Decay)',
    },
    4: {
        'data': 'result4.csv',
        'title': '$(p I + (1 - p) D)^{-\\frac{1}{2}} (A + qI) (p I + (1 - p) D)^{-\\frac{1}{2}}$\n'
                 + '(No LR Decay)\nI filled with 0 to match size of graph',
    },
}


def plot(i, exp):
    df = pd.read_csv(exp['data'], sep=';')

    ptc = df.loc[df['dataset'].str.startswith("PTC")]
    ptc['dataset'] = 'PTC'
    ptc['pretty_name'] = 'PTC'
    ptc = ptc.groupby(['dataset', 'pretty_name'], as_index=False).mean()

    df.drop(df.loc[df['dataset'].str.startswith("PTC")].index, inplace=True)
    df = df.append(ptc, sort=False)

    df.sort_values('pretty_name', inplace=True)

    xticks = np.arange(len(df))

    plt.title(exp['title'])
    plt.bar(xticks, df['mean_acc'], yerr=df['acc_std'])
    plt.plot(xticks, SOA.values(), 'r.')
    plt.xticks(xticks, df['pretty_name'])
    plt.ylim([0, 1])
    plt.savefig(f'exp3-result{i}.pdf')
    plt.show()


def main():
    for i, exp in exps.items():
        plot(i, exp)


if __name__ != 'main':
    main()
