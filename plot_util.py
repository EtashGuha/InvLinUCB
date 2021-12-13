from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

def plot(errors, times, title, Ts, NumArmss, remove_outliers=False):
    plt.clf()
    error_data = []
    time_data = []
    for T in Ts:
        for num_arms in NumArmss:
            for ele in errors[T][num_arms]:
                if not remove_outliers or ele < 20:
                    error_data.append((T, num_arms, ele))
                    
    for T in Ts:
        for num_arms in NumArmss:
            for ele in times[T][num_arms]:
                if not remove_outliers or ele < 20:
                    time_data.append((T, num_arms, ele))
                    
    x, y, z = zip(*error_data)
    _, _, times = zip(*time_data)
    print("Mean Error: {}, Mean Time: {}".format(np.mean(z), np.mean(times)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.scatter(x,y,z)
    ax.set_xlabel("T")
    ax.set_ylabel("Num Arms")
    ax.set_zlabel("MSE")
    plt.savefig("linucb.png")
    plt.show()

def plot_many(errors_list, names, title,  Ts, NumArmss, remove_outliers=False):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', "tab:red"]
    for idx, errors in enumerate(errors_list):
    
        data = []
        for T in Ts:
            for num_arms in NumArmss:
                for ele in errors[T][num_arms]:
                    if not remove_outliers or ele < 20:
                        data.append((T, num_arms, ele))

        x, y, z = zip(*data)
        ax.scatter(x,y,z, c=colors[idx], label=names[idx])
    ax.legend()
    ax.set_xlabel("T")
    ax.set_ylabel("Num Arms")
    ax.set_zlabel("MSE")
    plt.savefig("linucb.png")
    plt.show()