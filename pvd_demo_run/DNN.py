import torch
import numpy as np

class Stress_Model(torch.nn.Module):
    def __init__(self):
        super(Stress_Model, self).__init__()

        self.name = "Bayes_opt_demo"
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Linear(4, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
#             torch.nn.Tanh()
        )

    def forward(self, x_input):
        y_pred = self.network(x_input)
        return y_pred


class Stress_resistivity_Model(torch.nn.Module):
    def __init__(self):
        super(Stress_resistivity_Model, self).__init__()

        self.name = "Bayes_opt_demo"
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Linear(4, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 2),
            #             torch.nn.Tanh()
        )

    def forward(self, x_input):
        y_pred = self.network(x_input)
        return y_pred

    def regularization(self):
        l1_penalty = torch.nn.L1Loss()
        reg_loss = 0

        lam = 0
        for param in self.parameters():
            reg_loss += lam * l1_penalty(param, torch.zeros(param.shape))

        return reg_loss

    def loss(self, prediction, target):
        """_summary_

        Args:
            prediction (tensor):
            target (tensor):

        Returns:
            _type_: _description_
        """
        loss = torch.nn.MSELoss()
        return loss(prediction, target) + self.regularization()
