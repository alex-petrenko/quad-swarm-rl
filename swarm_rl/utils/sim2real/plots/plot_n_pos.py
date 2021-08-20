import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = 'n_pos/n_pos.txt' # about 14 seconds of data captured
    df = pd.read_csv(file)
    fig = plt.figure()
    nx, ny, nz = df['field.values0'].values, df['field.values1'].values, df['field.values2'].values

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(nx, ny, nz)
    plt.show()