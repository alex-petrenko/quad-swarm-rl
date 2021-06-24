import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    cf_ids = [7, 10, 11, 12]
    dfs = []
    file = 'swap_goals/swap_goals_data.txt'
    cfs_data = []
    plot_together = False
    df = pd.read_csv(file)
    fig = plt.figure()
    for cfid in cf_ids:
        cf_data = df.loc[df['field.transforms0.child_frame_id'] == f'cf{cfid}'][['field.transforms0.transform.translation.x',
                                                                 'field.transforms0.transform.translation.y',
                                                                 'field.transforms0.transform.translation.z']].values
        cfs_data.append(cf_data)

    if not plot_together:
        ax = fig.add_subplot(111, projection='3d')
        for cf_data in cfs_data:
            ax.plot(cf_data[:, 0], cf_data[:, 1], cf_data[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        for i in range(len(cf_ids)):
            cfid = cf_ids[i]
            cf_data = cfs_data[i]
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot(cf_data[:, 0], cf_data[:, 1], cf_data[:, 2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'cf{cfid}')
    plt.show()