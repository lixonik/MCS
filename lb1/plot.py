import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons

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