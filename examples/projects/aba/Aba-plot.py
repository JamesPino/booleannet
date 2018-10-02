"""
Absicis Acid Signaling - simulation

- plotting saved data

"""
import matplotlib.pyplot as plt
from boolean2 import util


def make_plot(fname):
    obj = util.bload(fname)
    data = obj['data']
    muts = obj['muts']
    genes = ['WT', 'pHc', 'PA']

    # standard deviations
    def limit(x):
        if x > 1:
            return 1
        elif x < 0:
            return 0
        else:
            return x

    plt.subplot(122)
    color = ['b', 'c', 'r']
    plots = []
    for gene, color, label in zip(genes, color, "WT pHc PA".split(),):
        means, std = data[gene]
        plt.plot(means, linestyle='-', color=color, label=label)
        upper = list(map(limit, means + std))
        lower = list(map(limit, means - std))
        plt.plot(upper, linestyle='--', color=color, lw=2)
        plt.plot(lower, linestyle='--', color=color, lw=2)

    plt.legend(loc='best')
    plt.title('Variability of Closure in WT and knockouts')
    plt.xlabel('Time Steps')
    plt.ylabel('Percent')
    plt.ylim((0.0, 1.1))
    # 
    # Plots the effect of mutations on Closure
    #
    plt.subplot(121)
    knockouts = 'WT S1P PA pHc ABI1'.split()

    for target in knockouts:
        plt.plot(muts[target]['Closure'], 'o-', label=target)
    plt.legend(loc='best')
    plt.title('Effect of mutations on Closure')
    plt.xlabel('Time Steps')
    plt.ylabel('Percent')
    plt.ylim((0, 1.1))


if __name__ == '__main__':
    plt.figure(num=None, figsize=(14, 7), dpi=80, facecolor='w', edgecolor='k')

    fname = 'ABA-run.bin'
    make_plot(fname)
    plt.show()
