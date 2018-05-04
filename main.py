import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from forager import Forager
from bee_colony import BeeColony

plt.ion()


filename = 'data/2.txt'
BC = BeeColony(filename, 100)

forager, path_len = BC.run(100, plot=1)


print(path_len)

