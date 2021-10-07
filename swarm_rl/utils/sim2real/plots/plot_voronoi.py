import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    cf_ids1 = [33, 41]
    cf_ids2 = [17, 20]
    cf_ids = cf_ids1 + cf_ids2
    dfs = []
    file1 = 'swap_voronoi/swap_voronoi_3341.txt'
    file2 = 'swap_voronoi/swap_voronoi_1720.txt'
    cfs_data = []
    plot_together = False
    df = pd.read_csv(file1)
    fig = plt.figure()
    for cfid in cf_ids1:
        cf_data = df.loc[df['field.transforms0.child_frame_id'] == f'cf{cfid}'][['field.transforms0.transform.translation.x',
                                                                 'field.transforms0.transform.translation.y',
                                                                 'field.transforms0.transform.translation.z']].values
        cfs_data.append(cf_data)


    df2 = pd.read_csv(file2)
    for cfid in cf_ids2:
        cf_data = df2.loc[df2['field.transforms0.child_frame_id'] == f'cf{cfid}'][['field.transforms0.transform.translation.x',
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