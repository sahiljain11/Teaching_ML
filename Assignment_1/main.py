# Assignmemt 1 - Univariate Linear Regression

import torch
import torchvision
import numpy
import random
import pandas as pd
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    # Constructor
    def __init__(self, file_name):
        # use pandas to read data from csv
        file_out = pd.read_csv(file_name)
        x = file_out['x'].values
        y = file_out['y'].values

        # convert the list into a torch tensor
        self.x = torch.tensor(x, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)

    # len(OBJECT_NAME)
    def __len__(self):
        return len(self.x)


data = FeatureDataset("data.csv")
m = len(data)

# 1) Compute the current loss given the following hypothesis function:
# h = theta_0 + theta_1 * x
# Hint: loss should be 8.3227 x 10^{9} (You may need to look up how to do summations,
# squares, etc in pytorch :P) Don't worry if you don't get the exact exact loss! It should
# be relatively close though
theta = torch.tensor([10, 100], dtype=torch.float64)

# 2) How about the following theta values?
# Hint: loss should be around 1.0224 x 10^{8}
theta = torch.tensor([35, -100], dtype=torch.float64)

# 3) Perform Gradient Descent to determine what the theta_0 and theta_1 values should be
# Note: Your theta[0] term may not move too much but I noticed that it changes extremely slowly
theta = torch.tensor([35, -100], dtype=torch.float64)
epochs = 100
step_size = .000005

# What were the values you got? What was your resulting loss? What happens if you increase
# your step_size? What are your theta values if your epoch was say ten thousand? A million?

# Congrats! You now know how to do linear regression using gradient descent!
# You are now well on your way to becoming a great ML Engineer!!! :D