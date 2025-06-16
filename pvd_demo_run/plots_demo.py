import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({'font.size': 22})

RESULT_DIR = "/home/ashriva/work/beyond_finger_printing/ankit/Experiments/Results/bayesopts/Bayes_opt_demo/GP_i15k1/"
NO_OLD_DATA = 78
stress_range = [-2000, 3000]
resist_range = [0, 30]
power_range = [50, 750]
pressure_range = [2, 23]
obj_range = [-.5, 1]

# Check whether the specified path exists or not
isExist = os.path.exists(RESULT_DIR + "/plots")

folders = ["/plots/3d_obj",
           "/plots/3d_resist",
           "/plots/3d_stress",
           "/plots/ucb",
           "/plots/stress_power",
           "/plots/stress_pressure",
           ]

for folder in folders:
    isExist = os.path.exists(RESULT_DIR + folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(RESULT_DIR + folder)


def matplotlib_3d_surface(indx, name, parameter_grid, val_array, val_obs, val_range):

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Experiment: " + str(indx))
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(parameter_grid[:, 0].reshape(xx.shape),
                    parameter_grid[:, 1].reshape(xx.shape),
                    val_array.reshape(yy.shape),
                    lw=0.5,
                    rstride=5, cstride=5,
                    edgecolors='k',
                    shade=False,
                    antialiased=True,
                    alpha=.4)

    ax.set_xlim(pressure_range[0], pressure_range[1])
    ax.set_ylim(power_range[0], power_range[1])
    ax.set_zlim(val_range[0], val_range[1])
    ax.set_xlabel('\n\nPressure (mTorr)', fontweight='bold')
    ax.set_ylabel('\n\nPower (W)', fontweight='bold')
    if(name == 'stress'):
        ax.set_zlabel("\n\n" + name + " (Mpa)", fontweight='bold')
    else:
        ax.set_zlabel("\n\n" + name, fontweight='bold')

    X_new = X_obs[NO_OLD_DATA:, :]
    X_old = X_obs[:NO_OLD_DATA, :]
    val_new = val_obs[NO_OLD_DATA:, :]
    val_old = val_obs[:NO_OLD_DATA, :]
    ax.scatter(X_old[:, 0],
               X_old[:, 1],
               val_old,
               color='r',
               label="observed")

    ax.scatter(X_new[:, 0],
               X_new[:, 1],
               val_new,
               color='g',
               label="new")

    scamap = plt.cm.ScalarMappable(cmap='jet')
    fcolors = scamap.to_rgba(val_array.reshape(yy.shape))
    # surf = ax.plot_surface(parameter_grid[:, 0].reshape(xx.shape),
    #                  parameter_grid[:, 1].reshape(xx.shape),
    #                  np.zeros(xx.shape)+val_range[0],
    #                 facecolors=fcolors, cmap='jet',
    #                 antialiased = True,
    #                 alpha=1,
    #                 linewidth=0.1)

    surf = ax.contourf(parameter_grid[:, 0].reshape(xx.shape),
                       parameter_grid[:, 1].reshape(xx.shape),
                       val_array.reshape(yy.shape),
                       zdir='z', offset=val_range[0],
                       cmap='jet')

    # contour = ax1.contour(xx, yy, val_array.reshape(xx.shape), 25,
    #                       vmax=val_range[1], vmin=val_range[0],
    #                       cmap=plt.cm.rainbow,
    #                       linestyles="solid", offset=val_range[0])
    cbar = plt.colorbar(surf, ax=ax, format='%.2f', fraction=0.01, pad=.1)

    fig.tight_layout()
    fig.legend()
    fig.savefig(RESULT_DIR + "/plots/3d_" + name + "/" + str(indx) + ".png")


def heatmap(indx):

    levels = np.arange(-0.4, 1.1, 0.02)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Experiment: " + str(indx))
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)
    ax2 = fig.add_subplot(1, 1, 1)
    # ax2 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=3)
    Z = ucb_array.reshape(xx.shape)
    contours = ax2.contourf(xx, yy, Z, cmap=plt.cm.hot_r)
                            #, levels=levels)
    # vmax=Z.max(), vmin=Z.min())
    cbar = fig.colorbar(contours, ax=ax2)
    # contours.set_clim(-0.5, 1.5)

    ax2.scatter(X_obs[:NO_OLD_DATA, 0], X_obs[:NO_OLD_DATA, 1], color='b',
                marker='o', label="observed data",
                s=200, clip_on=False)
    ax2.set_frame_on(False)
    ax2.set_xlabel("pressure (mTorr)")
    ax2.set_ylabel("Power (W)")
    ax2.set_title("Acquisition function (ucb)")
    ax2.set_xlim(pressure_range[0], pressure_range[1])
    ax2.set_ylim(power_range[0], power_range[1] + 5)
    ax2.grid(True)

    ax2.scatter(X_obs[NO_OLD_DATA:, 0], X_obs[NO_OLD_DATA:, 1], color='g',
                marker='o', label="new datapoints",
                s=200, clip_on=False)
    ax2.scatter(x_next[0, 0], x_next[0, 1], color='k',
                marker='x',
                s=200, linewidths=5, clip_on=False)
    ax2.annotate("next point",
                 xy=(x_next[0, 0] + .2, x_next[0, 1]), xycoords='data',
                 xytext=(x_next[0, 0] + 2, x_next[0, 1]), textcoords='data', size=30,
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3", lw=2))
    ax2.legend(loc='center left', bbox_to_anchor=(1.17, 0.5))
    # ax2.set_title("x: new data point     o: observed data points")

    # # two rows, two colums, combined third and fourth cell
    # ax3 = fig.add_subplot(2, 3, 4)
    # ax4 = fig.add_subplot(2, 3, 5)
    # ax5 = fig.add_subplot(2, 3, 6)
    # pr = []
    # po = []
    # aq = []
    # K_L = []
    # for k in range(indx + 1):
    #     x_next_ = np.load(RESULT_DIR + "/" + str(k) + "_x_next.npy")
    #     acq_next_ = np.load(RESULT_DIR + "/" + str(k) + "_acq_val_next.npy")
    #     pr.append(x_next_[0, 0])
    #     po.append(x_next_[0, 1])
    #     aq.append(acq_next_)
    #     K_L.append(k)
    # ax3.plot(K_L, pr, '-*')
    # ax4.plot(K_L, po, '-*')
    # ax5.plot(K_L, aq, '-*')

    # ax3.set_xlim(-0.5, 50)
    # ax4.set_xlim(-0.5, 50)
    # ax5.set_xlim(-0.5, 50)

    # [ax.set_xlabel('Iteration') for ax in [ax3, ax4, ax5]]
    # ax3.set_ylabel('Pressure')
    # ax4.set_ylabel('Power')
    # ax5.set_ylabel('Acq value')

    fig.tight_layout()
    # ax2.legend()
    plt.savefig(RESULT_DIR + "/plots/ucb/" + str(indx) + ".png")
    # plt.show()
    plt.close(fig)


def matplotlib_2d_pressure(k, parameter_grid):

    fig, axs = plt.subplots(3, 3, figsize=(25, 15))
    fig.suptitle("Experiment: " + str(k))
    X_new = X_obs[NO_OLD_DATA:, :]
    X_old = X_obs[:NO_OLD_DATA, :]
    stress_Y_new = stress_Y_obs[NO_OLD_DATA:, :]
    stress_Y_old = stress_Y_obs[:NO_OLD_DATA, :]
    resist_Y_new = resist_Y_obs[NO_OLD_DATA:, :]
    resist_Y_old = resist_Y_obs[:NO_OLD_DATA, :]
    obj_Y_new = obj_Y_obs[NO_OLD_DATA:, :]
    obj_Y_old = obj_Y_obs[:NO_OLD_DATA, :]

    # Plot GP_obj
    for i, p in enumerate([100, 500, 750]):
        # Plot GP_Stress
        indx = X_old[:, 1] == p
        axs[i, 0].scatter(X_old[indx, 0], stress_Y_old[indx],
                          color='r', label="observed data")

        indx_ = X_new[:, 1] == p
        axs[i, 0].scatter(X_new[indx_, 0], stress_Y_new[indx_],
                          color='g', label="new data")

        indx = np.isclose(parameter_grid[:, 1], p,
                          rtol=1e-05, atol=1e-08, equal_nan=False)
        axs[i, 0].plot(parameter_grid[indx, 0],
                       stress_mu_array[indx, 0], 'k')

        axs[i, 0].set_xlim(pressure_range[0],
                           pressure_range[1])
        axs[i, 0].set_ylim(stress_range[0],
                           stress_range[1])
        axs[i, 0].set_xlabel("pressure (mTorr)")
        axs[i, 0].set_ylabel("stress (Mpa)")
        axs[i, 0].set_title("Power: " + str(p) + " W")
        axs[i, 0].grid()
        axs[i, 0].legend()

        # Plot GP_resist
        # indx = X_old[:, 1] == p
        # axs[i, 1].scatter(X_old[indx, 0], resist_Y_old[indx],
        #                   color='r', label="observed data")

        # indx_ = X_new[:, 1] == p
        # axs[i, 1].scatter(X_new[indx_, 0], resist_Y_new[indx_],
        #                   color='g', label="new data")

        # indx = np.isclose(parameter_grid[:, 1], p,
        #                   rtol=1e-05, atol=1e-08, equal_nan=False)
        # axs[i, 1].plot(parameter_grid[indx, 0],
        #                resist_mu_array[indx, 0], 'k')

        # axs[i, 1].set_xlim(pressure_range[0],
        #                    pressure_range[1])
        # axs[i, 1].set_ylim(resist_range[0], resist_range[1])
        # axs[i, 1].set_xlabel("pressure (mTorr)")
        # axs[i, 1].set_ylabel("resistance (ohm/sq)")
        # axs[i, 1].set_title("Power: " + str(p) + " W")
        # axs[i, 1].grid()
        # axs[i, 1].legend()

        # Plot GP_obj
        pos = 1
        indx = X_old[:, 1] == p
        axs[i, pos].scatter(X_old[indx, 0],
                          obj_Y_old[indx],
                          color='r')
        indx_ = X_new[:, 1] == p
        axs[i, pos].scatter(X_new[indx_, 0],
                          obj_Y_new[indx_],
                          color='g')

        indx = np.isclose(parameter_grid[:, 1], p,
                          rtol=1e-05, atol=1e-08, equal_nan=False)

        axs[i, pos].plot(parameter_grid[indx, 0],
                       obj_mu_array[indx, 0],
                       'orange', label="GP fit")

        axs[i, pos].fill_between(parameter_grid[indx, 0],
                               obj_mu_array[indx, 0] + 2 *
                               obj_sd_array[indx, 0],
                               obj_mu_array[indx, 0] - 2 *
                               obj_sd_array[indx, 0],
                               alpha=0.5)
        # axs[i, 2].plot(parameter_grid[indx, 0],
        #                obj_func_array[indx, 0],
        #                "--k", label="Objective function")
        axs[i, pos].set_xlim(pressure_range[0],
                           pressure_range[1])
        axs[i, pos].set_ylim(obj_range[0], obj_range[1])
        axs[i, pos].set_xlabel("pressure (mTorr)")
        axs[i, pos].set_ylabel("obj")
        axs[i, pos].set_title("objective function")
        axs[i, pos].grid()
        axs[i, pos].legend()

        # UCB plot
        pos = 2
        axs[i, pos].plot(parameter_grid[indx, 0],
                       ucb_array[indx, 0],
                       'orange')
        axs[i, pos].set_xlim(pressure_range[0],
                           pressure_range[1])
        axs[i, pos].set_xlabel("pressure (mTorr)")
        axs[i, pos].set_ylabel("acq")
        axs[i, pos].set_ylim(obj_range[0], obj_range[1])
        axs[i, pos].set_title("Acquisition function (ucb)")
        axs[i, pos].grid()
        axs[i, pos].axvline(x=x_next[0, 0], ls='--', lw=2)

    fig.tight_layout()
    # plt.show()
    plt.savefig(RESULT_DIR + "/plots/stress_pressure/" + str(k) + ".jpg")
    plt.close(fig)


def matplotlib_2d_power(k, parameter_grid):

    X_new = X_obs[NO_OLD_DATA:, :]
    X_old = X_obs[:NO_OLD_DATA, :]
    stress_Y_new = stress_Y_obs[NO_OLD_DATA:, :]
    stress_Y_old = stress_Y_obs[:NO_OLD_DATA, :]
    resist_Y_new = resist_Y_obs[NO_OLD_DATA:, :]
    resist_Y_old = resist_Y_obs[:NO_OLD_DATA, :]
    obj_Y_new = obj_Y_obs[NO_OLD_DATA:, :]
    obj_Y_old = obj_Y_obs[:NO_OLD_DATA, :]

    fig, axs = plt.subplots(1, 5, figsize=(50, 10))
    axs = axs.reshape(1, -1)
    fig.suptitle("Experiment: " + str(k))
    # Plot GP_obj
    for i, p in enumerate([2]):
        # Plot GP_Stress
        indx = X_old[:, 0] == p
        axs[i, 0].scatter(X_old[indx, 1],
                          stress_Y_old[indx],
                          color='r', label="observed data", s=200)

        indx_ = X_new[:, 0] == p
        axs[i, 0].scatter(X_new[indx_, 1], stress_Y_new[indx_],
                          color='g', label="new data", s=200)

        indx = np.isclose(parameter_grid[:, 0], p,
                          rtol=1e-05, atol=1e-08, equal_nan=False)
        axs[i, 0].plot(parameter_grid[indx, 1],
                       stress_mu_array[indx, 0], 'k')

        axs[i, 0].set_xlim(power_range[0],
                           power_range[1])
        axs[i, 0].set_ylim(stress_range[0], stress_range[1])
        axs[i, 0].set_xlabel("Power (W)")
        axs[i, 0].set_ylabel("stress (Mpa)")
        axs[i, 0].set_title("Pressure: " + str(p) + " mTorr")
        axs[i, 0].grid()
        axs[i, 0].legend()

        # Plot GP_resist
        indx = X_old[:, 0] == p
        axs[i, 1].scatter(X_old[indx, 1], resist_Y_old[indx],
                          color='r', label="observed data", s=200)

        indx_ = X_new[:, 0] == p
        axs[i, 1].scatter(X_new[indx_, 1], resist_Y_new[indx_],
                          color='g', label="observed data")
        indx = np.isclose(parameter_grid[:, 0], p,
                          rtol=1e-05, atol=1e-08, equal_nan=False)
        axs[i, 1].plot(parameter_grid[indx, 1],
                       resist_mu_array[indx, 0], 'k')

        axs[i, 1].set_xlim(power_range[0],
                           power_range[1])
        axs[i, 1].set_ylim(resist_range[0], resist_range[1])
        axs[i, 1].set_xlabel("Power (W)")
        axs[i, 1].set_ylabel("resistance (ohm/sq)")
        axs[i, 1].set_title("Pressure: " + str(p) + " mTorr")
        axs[i, 1].grid()

        # GP_obj
        indx = X_old[:, 0] == p
        axs[i, 2].scatter(X_old[indx, 1],
                          obj_Y_old[indx],
                          color='r', s=200)

        indx_ = X_new[:, 0] == p
        axs[i, 2].scatter(X_new[indx_, 1],
                          obj_Y_new[indx_],
                          color='g', s=200)

        indexes = np.unique(X_new[indx_, 1], return_index=True)[1]
        uxnew = [X_new[indx_, 1][index] for index in sorted(indexes)]
        uynew = [obj_Y_new[indx_][index] for index in sorted(indexes)]
        for kk, (xx, yy) in enumerate(zip(uxnew, uynew)):
            axs[i, 2].annotate(str(kk),
                               xy=(xx + .2, yy + .01), fontsize=30)

        indx = np.isclose(parameter_grid[:, 0], p,
                          rtol=1e-05, atol=1e-08, equal_nan=False)

        axs[i, 2].plot(parameter_grid[indx, 1],
                       obj_mu_array[indx, 0],
                       'red', label="GP fit")

        axs[i, 2].fill_between(parameter_grid[indx, 1],
                               obj_mu_array[indx, 0] + 2 *
                               obj_sd_array[indx, 0],
                               obj_mu_array[indx, 0] - 2 *
                               obj_sd_array[indx, 0],
                               alpha=0.5)
        # axs[i, 2].plot(parameter_grid[indx, 1],
        #                obj_func_array[indx, 0],
        #                "--k", label="Objective function")
        axs[i, 2].set_xlim(power_range[0],
                           power_range[1])
        axs[i, 2].set_ylim(obj_range[0], obj_range[1])
        axs[i, 2].set_xlabel("Power (W)")
        axs[i, 2].set_ylabel("obj")
        axs[i, 2].set_title("objective function")
        axs[i, 2].grid()
        axs[i, 2].legend()

        # UCB plot
        axs[i, 3].plot(parameter_grid[indx, 1],
                       ucb_array[indx, 0],
                       'red')
        axs[i, 3].set_xlabel("Power (W)")
        axs[i, 3].set_ylabel("acq")
        axs[i, 3].set_xlim(power_range[0],
                           power_range[1])
        axs[i, 3].set_ylim(obj_range[0], obj_range[1])
        axs[i, 3].set_title("Acquisition function")
        axs[i, 3].grid()

        # grad plot
        axs[i, 4].plot(parameter_grid[indx, 1],
                       obj_array1[indx, 0],
                       '--k')
        axs[i, 4].set_xlabel("Power (W)")
        axs[i, 4].set_ylabel("grad")
        axs[i, 4].set_xlim(power_range[0],
                           power_range[1])
        axs[i, 4].set_title("Slope function")
        axs[i, 4].grid()

        if np.isclose(p, x_next[0, 0]):
            axs[i, 3].axvline(x=x_next[0, 1], ls='--', lw=2)
            axs[i, 3].annotate("next",
                               xy=(x_next[0, 1] + .2, 0), fontsize=30)
            axs[i, 4].axvline(x=x_next[0, 1], ls='--', lw=2)
            axs[i, 4].annotate("next",
                               xy=(x_next[0, 1] + .2, 0), fontsize=30)
            axs[i, 2].axvline(x=x_next[0, 1], ls='--', lw=2)
            axs[i, 2].annotate("next",
                               xy=(x_next[0, 1] + .2, 0), fontsize=30)

    fig.tight_layout()
    # plt.show()
    plt.savefig(RESULT_DIR + "/plots/stress_power/" + str(k) + ".jpg")
    plt.close(fig)


if __name__ == '__main__':

    num = int(sys.argv[1])

    for i in range(num):

        X_obs = np.load(RESULT_DIR + "/" + str(i) + "_x_obs.npy")
        x_next = np.load(RESULT_DIR + "/" + str(i) + "_x_next.npy")

        # Parameter grid setup
        xx = np.load(RESULT_DIR + "/" + str(i) + "_xgrid.npy")
        yy = np.load(RESULT_DIR + "/" + str(i) + "_ygrid.npy")
        parameter_grid = np.append(xx.reshape(-1, 1),
                                   yy.reshape(-1, 1),
                                   axis=1)

        # Stress Plots
        stress_mu_array = np.load(
            RESULT_DIR + "/" + str(i) + "_stress_mu_array.npy")
        stress_Y_obs = np.load(RESULT_DIR + "/" + str(i) + "_y_stress_obs.npy")

        matplotlib_3d_surface(i, 'stress', parameter_grid,
                              stress_mu_array, stress_Y_obs, stress_range)

        # Resist plots
        resist_mu_array = np.load(
            RESULT_DIR + "/" + str(i) + "_resist_mu_array.npy")
        resist_Y_obs = np.load(RESULT_DIR + "/" + str(i) + "_y_resist_obs.npy")

        matplotlib_3d_surface(i, 'resist', parameter_grid,
                              resist_mu_array, resist_Y_obs, resist_range)

        # Objective surface plots
        obj_func_array = np.load(
            RESULT_DIR + "/" + str(i) + "_obj_func_array.npy")
        obj_Y_obs = np.load(RESULT_DIR + "/" + str(i) + "_y_obj_obs.npy")

        matplotlib_3d_surface(i, 'obj', parameter_grid,
                              obj_func_array, obj_Y_obs, obj_range)

        # acq plots
        ucb_array = np.load(RESULT_DIR + "/" + str(i) + "_ucb_array.npy")
        acq_val_next = np.load(RESULT_DIR + "/" + str(i) + "_acq_val_next.npy")
        heatmap(i)

        # 2D plots
        obj_mu_array = np.load(RESULT_DIR + "/" + str(i) + "_obj_mu_array.npy")
        obj_sd_array = np.load(RESULT_DIR + "/" + str(i) + "_obj_sd_array.npy")
        obj_array1 = np.load(RESULT_DIR + "/" + str(i) + "_obj_array1.npy")
        obj_array1 = (-obj_array1 + 1) * 1900

        matplotlib_2d_power(i, parameter_grid)
        matplotlib_2d_pressure(i, parameter_grid)
