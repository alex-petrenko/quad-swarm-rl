import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    cf_ids = [12]
    dfs = []
    for cfid in cf_ids:
        file = f'swap_goals/cf{cfid}.txt'
        df = pd.read_csv(file)
        dfs.append(df)
        print(type(df['field.values1']))
        print(list(df['field.values1']))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for df in dfs:
        xs = list(df['field.values0'])[411:]
        ys = list(df['field.values1'])[411:]
        zs = list(df['field.values2'])[411:]
        ax.plot(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()