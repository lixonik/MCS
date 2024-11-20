import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from typing import Callable, List

class Euler:
    @classmethod
    def integrate(cls, f: Callable, x: List[int], h: int) -> List[int]:
        dx = f(x)
        return x + dx * h

class Sprott():
    @property
    def initConditions(self):
        return (0.1, 0, 0)
    
    @property
    def stepSize(self):
        return 0.1

    @classmethod
    def calculate(self, xyz: list[float]) -> list[float]:
        x, y, z = xyz
        return np.array([-0.2 * y, 
                x + z, 
                x + y**2 - z])
    
    @classmethod
    def jacobian(self, xyz: list[float]) -> list[float]:
        y = xyz[1]
        return np.array([[0, -0.2, 0], 
                         [1, 0, 1], 
                         [1, 2*y, -1]])

class KalmanFilter:
    def __init__(self, processNoise, measurementNoise, init_state) -> None:
        # Filter parameters
        self._state = init_state.reshape(3, 1) # initial state [x, y, z]
        
        self._F = np.eye(3) # transition matrix
        # self._F = Lorenz.jacobian(self._state)
        self._H = np.eye(3) # observation matrix
        self._P = np.eye(3) # covariance matrix

        self._Q = np.eye(3) * processNoise  # covariance of the process noise
        self._R = np.eye(3) * measurementNoise  # covariance of the observation noise
        self._ODE = Sprott()
    
    def predict(self) -> None:
        # x = F*x
        # P = F*P*F.T + Q
        
        # Predict new state
        self._state = np.dot(self._F, self._state)
        # self._state = Euler.integrate(self._ODE.calculate, self._state, self._ODE.stepSize)
        
        # Predict covariance fault
        self._P = np.dot(self._F, np.dot(self._P, self._F.T)) + self._Q
        
    def correct(self, measurement):
        # y = z - H*x
        # S = H*P*H.T + R
        # K = P*H*S^(-1)
        # x = x + K*y
        # P = (I - K*H)*P
        
        # Make a measurement column-like
        measurement = measurement.reshape(3, 1)

        # Calculate Kalman gain matrix
        S = np.dot(self._H, np.dot(self._P, self._H.T)) + self._R  # observating fault
        K = np.dot(self._P, np.dot(self._H.T, np.linalg.inv(S)))  # Kalman gain matrix
        
        # Calculate the remainder
        y = measurement - np.dot(self._H, self._state)  # measuring error
        # Update the state
        self._state = self._state + np.dot(K, y)
        # Update the covariance fault
        I = np.eye(self._P.shape[0])  # unit matrix of dimention as P
        self._P = (I - np.dot(K, self._H)).dot(self._P)
    
    @property
    def state(self) -> np.array:
        return self._state.flatten()
      
def plot(noisyData, filteredData, simulatedData, xRange, label1, label2):
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    lineNoisy2d1, = ax1.plot(noisyData.T[0], noisyData.T[1], "r-", label='data')
    lineFiltered2d1, = ax1.plot(filteredData.T[0], filteredData.T[1], "b-", label='filtred data')
    lineSimulated2d1, = ax1.plot(simulatedData.T[0], simulatedData.T[1], "g-", label='simulation')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(label1)

    # Right 2d subplot
    ax2 = fig.add_subplot(122)
    lineNoisy2d2, = ax2.plot(xRange, noisyData.T[0], "r-", label='data')
    lineFiltered2d2, = ax2.plot(xRange, filteredData.T[0], "b-", label='filtred data')
    lineSimulated2d2, = ax2.plot(xRange, simulatedData.T[0], "g-", label='simulation')

    ax2.set_xlabel("t")
    ax2.set_ylabel("θ")
    ax2.set_title(label2)

    # Select x-t/y-t/z-t buttons
    radio_ax = plt.axes([0.87, 0.125, 0.1, 0.08])
    radio = RadioButtons(radio_ax, ('x over time', 'y over time', 'z over time'))

    def labelFunc(label):
        if label == 'x over time':
            lineNoisy2d2.set_ydata(noisyData.T[0])
            lineFiltered2d2.set_ydata(filteredData.T[0])
            lineSimulated2d2.set_ydata(simulatedData.T[0])
            ax2.set_ylabel("θ")
        elif label == 'y over time':
            lineNoisy2d2.set_ydata(noisyData.T[1])
            lineFiltered2d2.set_ydata(filteredData.T[1])
            lineSimulated2d2.set_ydata(simulatedData.T[1])
            ax2.set_ylabel("θ")
        elif label == 'z over time':
            lineNoisy2d2.set_ydata(noisyData.T[2])
            lineFiltered2d2.set_ydata(filteredData.T[2])
            lineSimulated2d2.set_ydata(simulatedData.T[2])
            ax2.set_ylabel("θ")
        
        ax2.relim()
        ax2.autoscale_view()
        plt.draw()

    radio.on_clicked(labelFunc)
    
    fig.legend(handles=[lineNoisy2d1, lineFiltered2d1, lineSimulated2d1], loc='lower center')
    
    # Select visible graph checkboxes
    checkbox_ax = plt.axes([0.0, 0.0, 0.12, 0.2])
    checkbox = CheckButtons(checkbox_ax, 
                            ['data', 'filtred data', 'simulation'], 
                            [True, True, False],
                            frame_props={'edgecolor': ['red', 'blue', 'green']},
                            check_props={'facecolor': ['red', 'blue', 'green']})
    lineSimulated2d1.set_visible(False)
    lineSimulated2d2.set_visible(False)
    
    def checkboxFunc(label):
        if label == 'data':
            lineNoisy2d1.set_visible(not lineNoisy2d1.get_visible())
            lineNoisy2d2.set_visible(not lineNoisy2d2.get_visible())
        elif label == 'filtred data':
            lineFiltered2d1.set_visible(not lineFiltered2d1.get_visible())
            lineFiltered2d2.set_visible(not lineFiltered2d2.get_visible())
        elif label == 'simulation':
            lineSimulated2d1.set_visible(not lineSimulated2d1.get_visible())
            lineSimulated2d2.set_visible(not lineSimulated2d2.get_visible())

        plt.draw()

    checkbox.on_clicked(checkboxFunc)

    plt.tight_layout()
    plt.show()

ODE = Sprott()
INTEGRATOR = Euler()

def main():  
    # Getting noisy measurments from a file
    noisyData = []
    path = "../data/data3.txt"
        
    with open(path, "r") as file:
        for line in file:
            noisyData.append([float(x) for x in line.split()])
    noisyData = np.array(noisyData)
    
    # System simulation
    step = ODE.stepSize
    num_steps = len(noisyData)

    simulatedData = np.empty((num_steps, 3))
    simulatedData[0] = ODE.initConditions

    for i in range(num_steps - 1):
        simulatedData[i + 1] = INTEGRATOR.integrate(ODE.calculate, simulatedData[i], step)
        
    # KF usage
    processNoise = 1e-3
    measurementNoise = 0.02
    initState = np.array([0.1, 0.1, 0.1])

    # Filter initalization
    kf = KalmanFilter(processNoise, measurementNoise, initState)

    # Filtering process
    filteredData = []
    for measurement in noisyData:
        kf.predict()
        kf.correct(measurement)
        filteredData.append(kf.state)

    filteredData = np.array(filteredData)

    # Gather parameters for view
    xRange = []
    for i in range(num_steps):
        xRange.append(ODE.stepSize * i)
        
    label1 = "Фазовый портрет"
    label2 = "Временная диаграмма"
        
    # Visualization
    plot(noisyData, filteredData, simulatedData, xRange, label1, label2)
    
if __name__ == '__main__':
    main()