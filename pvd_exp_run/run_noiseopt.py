import os
import numpy as np
import warnings
import sys
import pickle
sys.path.append('../')

from datasetup import stress_resist_data_loader, append_new_data
from src.optimizers import tune_hyperparam_parallel
from src.bayesopt import Bayesopt
from src.gaussian_process import Gaussain_Process
from src.acquisition_func import ucb
from src.kernels import rbf
warnings.filterwarnings("ignore")


RESULT_DIR = "/home/ashriva/work/Codes/bayesopt/Results/run_opt5"

isExist = os.path.exists(RESULT_DIR)
if not isExist:
    os.makedirs(RESULT_DIR)

# required for scaling don't change
# (min max) pairs in each dimension
parameter_space = np.array([[2, 23], [50, 750]])
# theta_space


def stress_obj(stress, stress_grad, resistivity):

    #     def Relu(x):
    #         return x * (x > 0)

    def modified_sigmoid(x, d):

        x = d - x
        return 1 / (1 + np.exp(-5 * x))

    def cutoff_func(x, d):

        def sigmoid(x):

            return 1 / (1 + np.exp(-x))

        f = sigmoid(d + x) + sigmoid(d - x)
        f = (f - 1)

        return f

    def pos_grad_func(x):

        return (np.tanh(x) + 1) / 2

    def mingrad_func(x):

        return (-x / 1900 + 1)

    min_grad_criteria = mingrad_func(stress_grad)
    pos_grad_criteria = pos_grad_func(stress_grad)
    stress_cutoff = cutoff_func(stress, 300)
    resistivity_cutoff = modified_sigmoid(resistivity, 3)

    switch = pos_grad_criteria * stress_cutoff * resistivity_cutoff
    obj = min_grad_criteria * switch

    return obj, min_grad_criteria, pos_grad_criteria, stress_cutoff, resistivity_cutoff


def save_results(k, GP_obj, GP_stress, GP_resist, x_next, acq_val, obj_func, acq_func):

    with open(RESULT_DIR + "/" + str(k) + '_gp_objects.pkl', 'wb') as outp:
        data = {'stress': GP_stress,
                'resist': GP_resist,
                'obj': GP_obj}
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    # Plot surface
    stress_mu_array = np.zeros([0, 1])
    stress_sd_array = np.zeros([0, 1])
    resist_mu_array = np.zeros([0, 1])
    resist_sd_array = np.zeros([0, 1])

    obj_mu_array = np.zeros([0, 1])
    obj_sd_array = np.zeros([0, 1])
    obj_func_array = np.zeros([0, 1])
    ucb_array = np.zeros([0, 1])

    obj_array1 = np.zeros([0, 1])
    obj_array2 = np.zeros([0, 1])
    obj_array3 = np.zeros([0, 1])
    obj_array4 = np.zeros([0, 1])

    xx = np.linspace(parameter_space[0][0], parameter_space[0][1], 100)
    xx = np.append(xx, [2, 3, 5, 8, 11, 14, 17, 20, 23, x_next[0, 0]])
    xx = np.unique(xx)

    yy = np.linspace(parameter_space[1][0], parameter_space[1][1], 20)
    yy = np.append(yy, [100, 500, 750, x_next[0, 1]])
    yy = np.unique(yy)

    # Parameter grid setup
    xx, yy = np.meshgrid(xx, yy)
    parameter_grid = np.append(xx.reshape(-1, 1),
                               yy.reshape(-1, 1),
                               axis=1)

    for input_ in parameter_grid:
        input_ = input_.reshape(1, -1)

        # Objective surface seen by the GP_obj
        mu_, sd_ = GP_obj.posterior(input_)

        obj_sd_array = np.append(obj_sd_array,
                                 sd_.reshape(-1, 1),
                                 axis=0)
        obj_mu_array = np.append(obj_mu_array,
                                 mu_.reshape(-1, 1),
                                 axis=0)
        ucb_val = acq_func(input_).reshape(-1, 1)
        ucb_array = np.append(ucb_array, ucb_val, axis=0)

        # Objective surface seen by the GP_stress
        stress_mu_, stress_sd_, stress_mu_grad_ = GP_stress.posterior(
            input_, calc_grad=True)

        resist_mu_, resist_sd_ = GP_resist.posterior(input_)

        obj_val, obj1, obj2, obj3, obj4 = obj_func(
            stress_mu_, stress_mu_grad_, resist_mu_)

        obj_func_array = np.append(obj_func_array, obj_val, axis=0)
        obj_array1 = np.append(obj_array1, obj1.reshape(-1, 1), axis=0)
        obj_array2 = np.append(obj_array2, obj2.reshape(-1, 1), axis=0)
        obj_array3 = np.append(obj_array3, obj3.reshape(-1, 1), axis=0)
        obj_array4 = np.append(obj_array4, obj4.reshape(-1, 1), axis=0)

        stress_sd_array = np.append(
            stress_sd_array, stress_sd_.reshape(-1, 1), axis=0)
        resist_sd_array = np.append(
            resist_sd_array, resist_sd_.reshape(-1, 1), axis=0)
        stress_mu_array = np.append(
            stress_mu_array, stress_mu_.reshape(-1, 1), axis=0)
        resist_mu_array = np.append(
            resist_mu_array, resist_mu_.reshape(-1, 1), axis=0)

    # Save data
    np.save(RESULT_DIR + "/" + str(k) + "_xgrid", xx)
    np.save(RESULT_DIR + "/" + str(k) + "_ygrid", yy)
    np.save(RESULT_DIR + "/" + str(k) + "_stress_mu_array", stress_mu_array)
    # np.save(RESULT_DIR + "/" + str(k) + "_stress_sd_array", stress_sd_array)
    # np.save(RESULT_DIR + "/" + str(k) + "_resist_mu_array", resist_mu_array)
    # np.save(RESULT_DIR + "/" + str(k) + "_resist_sd_array", resist_sd_array)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_mu_array", obj_mu_array)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_sd_array", obj_sd_array)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_func_array", obj_func_array)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_array1", obj_array1)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_array2", obj_array2)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_array3", obj_array3)
    np.save(RESULT_DIR + "/" + str(k) + "_obj_array4", obj_array4)
    np.save(RESULT_DIR + "/" + str(k) + "_ucb_array", ucb_array)
    np.save(RESULT_DIR + "/" + str(k) + "_x_next", x_next)
    np.save(RESULT_DIR + "/" + str(k) + "_acq_val_next", acq_val)
    np.save(RESULT_DIR + "/" + str(k) + "_x_obs", GP_stress.X_obs)
    np.save(RESULT_DIR + "/" + str(k) + "_y_stress_obs", GP_stress.Y_obs)
    np.save(RESULT_DIR + "/" + str(k) + "_y_resist_obs", GP_resist.Y_obs)
    np.save(RESULT_DIR + "/" + str(k) + "_y_obj_obs", GP_obj.Y_obs)
    np.save(RESULT_DIR + "/" + str(k) +
            "_theta_stress", GP_stress.kernel.theta)
    np.save(RESULT_DIR + "/" + str(k) +
            "_theta_resist", GP_resist.kernel.theta)
    np.save(RESULT_DIR + "/" + str(k) + "_theta_obj", GP_obj.kernel.theta)


if __name__ == '__main__':

    if(sys.argv[1] not in ["-man", "-opt", "-bopt_only"]):
        print("Invalid argument")
        exit()

    if sys.argv[1] == "-man":
        opt = False
        ls1 = float(sys.argv[2])
        ls2 = float(sys.argv[3])
        run = sys.argv[4]

        # [sigma, l_pressure, l_power]
        l_stress = [1, 0.3, 0.45]
        l_resist = l_stress
        l_obj = [1, ls1, ls2]

    elif sys.argv[1] == "-opt" or sys.argv[1] == "-bopt_only":
        opt = True
        l_stress = [1, 0.3, 0.45]
        l_resist = l_stress
        l_obj = l_stress
        run = sys.argv[2]

        trial = int(run)
        print("Iteration %d" % trial)

    # Data setup
    if(trial == 0):
        x_train, stress_y_train, resist_y_train = stress_resist_data_loader()

    else:
        x_train = np.load(RESULT_DIR + "/" + str(trial - 1) + "_x_obs.npy")
        stress_y_train = np.load(
            RESULT_DIR + "/" + str(trial - 1) + "_y_stress_obs.npy")
        resist_y_train = np.load(
            RESULT_DIR + "/" + str(trial - 1) + "_y_resist_obs.npy")

        x_new, stress_new, resist_new = append_new_data(trial - 1)

        x_train = np.append(x_train, x_new, axis=0)
        stress_y_train = np.append(
            stress_y_train, stress_new.reshape(-1, 1), axis=0)
        resist_y_train = np.append(
            resist_y_train, resist_new.reshape(-1, 1), axis=0)

    if(sys.argv[1] != "-bopt_only"):
        # Gaussian process interpolator for stress
        kernel_stress = rbf(l_stress)
        GP_stress = Gaussain_Process(kernel_stress, parameter_space)
        GP_stress.add_observation(x_train, stress_y_train, scale=True)

        if opt:
            #GP_stress.fit()
            theta = tune_hyperparam_parallel(GP_stress)
            GP_stress.kernel.theta = theta[1:]
            GP_stress.noise = theta[0]
        else:
            score = GP_stress.negative_log_likelihood(GP_stress.kernel.theta)
            print("manual:\t ", score, GP_stress.kernel.theta)

        # # Gaussian process interpolator for resistance
        kernel = rbf(l_resist)
        GP_resist = Gaussain_Process(kernel, parameter_space)
        GP_resist.add_observation(x_train, resist_y_train, scale=True)

        if opt:
            #GP_resist.fit()
            theta = tune_hyperparam_parallel(GP_resist)
            GP_resist.kernel.theta = theta[1:]
            GP_resist.noise = theta[0]
        else:
            score = GP_resist.negative_log_likelihood(GP_resist.kernel.theta)
            print("manual:\t ", score, GP_resist.kernel.theta)

        # Create Gausian process for desired objective function
        kernel = rbf(l_obj)
        GP_obj = Gaussain_Process(kernel, parameter_space)
        stress_mu_, _, stress_mu_grad_ = GP_stress.posterior(
            x_train, calc_grad=True)
        resist_mu_, _ = GP_resist.posterior(x_train)
        obj_val, _, _, _, _ = stress_obj(
            stress_y_train, stress_mu_grad_, resist_y_train)
        GP_obj.add_observation(x_train, obj_val, scale=True)

        if opt:
            theta = tune_hyperparam_parallel(GP_obj)
            GP_obj.kernel.theta = theta[1:]
            GP_obj.noise = theta[0]
        else:
            score = GP_obj.negative_log_likelihood(GP_obj.kernel.theta)
            print("manual:\t ", score, GP_obj.kernel.theta)

    else:
        with open(RESULT_DIR + "/" + str(trial) + '_gp_objects.pkl', 'rb') as file:
            data = pickle.load(file)
            GP_stress = data['stress']
            GP_resist = data['resist']
            GP_obj = data['obj']

    # Bayesian optimization ------------------------------------------------
    # Define Bayesopt
    bopt = Bayesopt()
    search_bounds = parameter_space
    acq_func = ucb(GP_obj)

    for k in range(1, 10):
        acq_func.k = k

        # Bayesopt
        # Suggest next point
        x_next, acq_val = bopt.next_point(acq_func, search_bounds)
        print((x_next[0], round(x_next[1], 3)), acq_val)
    x_next = x_next.reshape(1, -1)
    # save_results(trial,
    #              GP_obj, GP_stress, GP_resist,
    #              x_next, acq_val,
    #              stress_obj, acq_func)
