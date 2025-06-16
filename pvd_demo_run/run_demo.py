import os
import re
import torch
import numpy as np
import pandas as pd
import warnings
import sys
import random
import pickle
sys.path.append('../../')
warnings.filterwarnings("ignore")
random.seed(10)

from DNN import Stress_resistivity_Model as Model
from src.functions import latin_hypercube_2d_uniform
from src.kernels import rbf
from src.acquisition_func import ucb
from src.gaussian_process import Gaussain_Process
from src.bayesopt import Bayesopt
from src.optimizers import tune_hyperparam_parallel
from stress_optimisation.load_exp_data import stress_resist_data_loader, append_new_data

explore_K = 1
Ninit_samples = 15
RESULT_DIR = "/home/ashriva/work/beyond_finger_printing/ankit/Experiments/Results/bayesopts/Bayes_opt_demo/GP_i"+str(Ninit_samples)+"k"+str(explore_K)+"/"

isExist = os.path.exists(RESULT_DIR)
if not isExist:
    os.makedirs(RESULT_DIR)

# required for scaling don't change
# (min max) pairs in each dimension
parameter_space = np.array([[2, 23], [50, 750]])

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
    np.save(RESULT_DIR + "/" + str(k) + "_stress_sd_array", stress_sd_array)
    np.save(RESULT_DIR + "/" + str(k) + "_resist_mu_array", resist_mu_array)
    np.save(RESULT_DIR + "/" + str(k) + "_resist_sd_array", resist_sd_array)
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

class NN_surrogate_model:

    def __init__(self):

        self.model = Model()
        self.model.load_state_dict(torch.load("DNN_combine.pth"))
        self.model.eval()

        with open('demo_gp_objects.pkl', 'rb') as file:
            data = pickle.load(file)
            self.gp_stress = data['stress']
            self.gp_resist = data['resist']

    def __call__(self, x):

        x = self.gp_stress.get_scale_x(x)
        x = x.astype(np.float32)
        x = torch.from_numpy(x.reshape(-1, 2))
        y = self.model(x).detach().numpy()
        y += .1 * np.random.normal(0, 1, size=y.shape)

        y1 = self.gp_stress.invscale_y(y[:, 0].reshape(-1, 1))
        y2 = self.gp_resist.invscale_y(y[:, 1].reshape(-1, 1))

        return y1, y2

class GP_surrogate_model:

    def __init__(self):

        with open('demo_gp_objects.pkl', 'rb') as file:
            data = pickle.load(file)
            self.gp_stress = data['stress']
            self.gp_resist = data['resist']

    def __call__(self, x):

        y1, _ = self.gp_stress.posterior(x)
        y1 += .1 * np.random.normal(0, 1, size=y1.shape)

        y2, _ = self.gp_resist.posterior(x)
        y2 += .1 * np.random.normal(0, 1, size=y2.shape)

        return y1, y2

if __name__ == '__main__':

    # [sigma, l_pressure, l_power]
    l_stress = [1, 0.3, 0.45]
    l_resist = l_stress
    l_obj = l_resist

    # Gaussian process interpolator for stress
    kernel_stress = rbf(l_stress)
    GP_stress = Gaussain_Process(kernel_stress, parameter_space)
    GP_stress.noise = 0.1

    # # Gaussian process interpolator for resistance
    kernel_resist = rbf(l_resist)
    GP_resist = Gaussain_Process(kernel_resist, parameter_space)
    GP_resist.noise = 0.1

    # Create Gausian process for desired objective function
    kernel_obj = rbf(l_obj)
    GP_obj = Gaussain_Process(kernel_obj, parameter_space)
    GP_obj.noise = 1e-3

    # Create blackbox function for quering stress
    blackbox_func = GP_surrogate_model()

    # Define Bayesopt
    bopt = Bayesopt()
    search_bounds = parameter_space
    acq_func = ucb(GP_obj)
    acq_func.k = explore_K

    # Run experiments ------------------------------------------------

    ## LHS sampling
    x_train = latin_hypercube_2d_uniform(
        parameter_space[0], parameter_space[1], Ninit_samples)
    stress_y_train, resist_y_train = blackbox_func(x_train)
    print(x_train, parameter_space[0], parameter_space[1])

    ## Random sampling
    x_train, stress_y_train, resist_y_train = stress_resist_data_loader()
    indx = random.sample(range(x_train.shape[0]), 15)
    x_train = x_train[indx]
    print(x_train)
    exit()
    # stress_y_train = stress_y_train[indx]
    # resist_y_train = resist_y_train[indx]

    for trial in range(10):
        print("Iteration %d" % trial)

        GP_stress.reset()
        GP_stress.add_observation(x_train, stress_y_train, scale=True)
        GP_stress.fit()

        GP_resist.reset()
        GP_resist.add_observation(x_train, resist_y_train, scale=True)
        GP_resist.fit()

        GP_obj.reset()
        _, _, stress_mu_grad_ = GP_stress.posterior(x_train, calc_grad=True)
        _, _ = GP_resist.posterior(x_train)
        obj_val, _, _, _, _ = stress_obj(stress_y_train, stress_mu_grad_, resist_y_train)
        GP_obj.add_observation(x_train, obj_val, scale=True)
        GP_obj.fit()

        # Bayesopt
        # Suggest next point
        x_next, acq_val = bopt.next_point(acq_func, search_bounds)
        print((x_next[0], round(x_next[1], 3)), acq_val)
        x_next = x_next.reshape(1, -1)
        
        save_results(trial,
                    GP_obj, GP_stress, GP_resist,
                    x_next, acq_val,
                    stress_obj, acq_func)

        # Query data from the models
        stress_next, resist_next = blackbox_func(x_next)
        stress_next = stress_next.reshape(1, -1)
        resist_next = resist_next.reshape(1, -1)

        # Update to existing data
        x_train = np.append(x_train, x_next, axis=0)
        resist_y_train = np.append(resist_y_train, stress_next, axis=0)
        stress_y_train = np.append(stress_y_train, stress_next, axis=0)
    print("\n")
