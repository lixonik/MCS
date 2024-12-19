from Sprott import Sprott
from Euler import Euler
from KalmanFilter import KalmanFilter
from plot import plot
import numpy as np

ODE = Sprott()
INTEGRATOR = Euler()

def main():  
    # Getting noisy measurments from a file
    noisyData = []
    path = "./data/data3.txt"
        
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
    kf = KalmanFilter(processNoise, measurementNoise, initState, ODE, INTEGRATOR)

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
        
    label1 = "Фазовая область решения"
    label2 = "Временная область решения"
        
    # Visualization
    plot(noisyData, filteredData, simulatedData, xRange, label1, label2)
    
if __name__ == '__main__':
    main()