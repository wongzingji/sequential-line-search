import pySequentialLineSearch
import numpy as np
import sys
from typing import Tuple
from PyQt5.QtWidgets import QApplication
sys.path.append('./')
from gui.window import UIWidget
import cv2
import time

PARAMS = {'w1':[0,10,1], 'w2':[0,100,10], 'w3':[0,50,5]} # {name:range [start,end,interval]}
NUM_DIMS = len(PARAMS)
IMG_SIZE = 64
CHANNELS = 3

# obj func: loss func
def img_objective_func(img_arr: np.ndarray, weights: np.ndarray, f: float) -> float:
    # mask = np.repeat(np.tile(np.linspace(1, 0, img_arr.shape[1]), (img_arr.shape[0], 1))[:, :, np.newaxis], 3, axis=2) 
    blurred = cv2.GaussianBlur(img_arr, (0, 0), 3)
    enhanced = cv2.subtract(img_arr, blurred)
    enhanced = enhanced * f
    print(enhanced.dtype)
    perception = cv2.addWeighted(img_arr, weights[0], enhanced, weights[1], 0, dtype = cv2.CV_32F)
    perception = perception.astype(np.uint8)
    return perception

def calc_simulated_objective_func(params: np.ndarray):    
    # return -np.linalg.norm(x - 0.2)
    
    x = np.linspace(0, 10, 100)
    y = np.cos(x + params[1] * np.pi / 180) * np.exp(-params[0] * x) * params[2]
    return -y

# A dummy implementation of slider manipulation
def ask_human_for_slider_manipulation(
        slider_ends: Tuple[np.ndarray, np.ndarray]):
    t_max = 0.0
    f_max = -sys.float_info.max

    for i in range(1000):
        t = float(i) / 999.0
        x = (1.0 - t) * slider_ends[0] + t * slider_ends[1]
        f = np.max(calc_simulated_objective_func(x))

        if f_max is None or f_max < f:
            f_max = f
            t_max = t

    return t_max


# A custom generator of a slider for the first iteration
def generate_initial_slider(num_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    end_0 = np.random.uniform(low=0.0, high=1.0, size=(num_dims, ))
    end_1 = np.random.uniform(low=0.0, high=1.0, size=(num_dims, ))
    return end_0, end_1


# An implementation of sequential line search procedure
def main():
    # init optimizer
    optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
        num_dims=NUM_DIMS,
        use_map_hyperparams=True,
        acquisition_func_type=pySequentialLineSearch.AcquisitionFuncType.ExpectedImprovement, # EI
        initial_query_generator=generate_initial_slider)

    optimizer.set_hyperparams(kernel_signal_var=0.50,
                              kernel_length_scale=0.10,
                              kernel_hyperparams_prior_var=0.10)

    # init app
    app = QApplication(sys.argv)
    w = UIWidget(calc_simulated_objective_func, PARAMS, optimizer, ask_human_for_slider_manipulation)
    w.show()

    time.sleep(10)
    for i in range(15):
        w.update_window()
        time.sleep(1)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
