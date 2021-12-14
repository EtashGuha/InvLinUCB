from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

def plot(errors, title, name, folder, remove_outliers=False):
    plt.clf()
    error_data = []
    for T in errors.keys():
        for num_arms in errors[T].keys():
            for ele in errors[T][num_arms]:
                    error_data.append((T, num_arms, ele))
                    
    x, y, z = zip(*error_data)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.scatter(x,y,z)
    ax.set_xlabel("T")
    ax.set_ylabel("Num Arms")
    ax.set_zlabel("MSE")
    plt.savefig("{}/{}.png".format(folder, name))

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