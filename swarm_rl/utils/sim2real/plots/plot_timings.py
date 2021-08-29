import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = 'timings/timers.txt' # about 14 seconds of data captured
    df = pd.read_csv(file)
    fig = plt.figure()
    ekf_start, ekf_end = df['field.values0'].values, df['field.values1'].values
    ctrl_start, ctrl_end = df['field.values2'].values, df['field.values3'].values

    ax = fig.add_subplot(111)
    xrange = list(range(0, ekf_start.shape[0]))
    ax.plot(xrange, ekf_end - ekf_start, label='ekf')
    ax.plot(xrange, ctrl_end - ctrl_start, label='nn')
    ax.set_ylabel('runtime (usec)')
    plt.legend()
    plt.show()