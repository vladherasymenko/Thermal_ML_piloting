import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# TODO Add docstrings to all methods. Explain private methods. 
# TODO Add validation and error handling 

class Generator: 
    """
    Simulates and generates test data for a thermal system.
    """
    def __init__(self, save_rate, 
                 m = 50.0, cp = 200.0, h_base = 10.0, 
                 s = 3, T_amb = 300.0, P_max = 20000, 
                 T_initial = 300.0, N_t_points = 3000):
        
        self.save_rate = save_rate    # How often the data points are saved, e.g. save_rate = 10 - every 10 seconds
        self.m = m                    # mass in kg
        self.cp = cp                  # specific heat capacity in J/(kg·K)
        self.h_base = h_base          # base convective heat transfer coefficient in W/(m^2·K)
        self.s = s                    # surface area in m^2
        self.T_amb = T_amb            # ambient temperature in K
        self.P_max = P_max            # maximum power input in watts
        self.T_initial = T_initial    # initial temperature in K
        self.N_t_points = N_t_points  # number of data points to generate 


    def _test_power_input(self, x, t): # TODO remake with array, not function

        if t < 500: # TODO make these values dependent on self.N_t_points
            return self.P_max * (0.3+x)
        elif 500 <= t < 1000:
            return self.P_max * (0.4+x)
        elif 1000 <= t < 1500:
            return self.P_max * (0.5+x)
        elif 1500 <= t < 2000:
            return self.P_max * (0.6+x)
        else:
            return 0

    def generate_test(self, file_name, n_curves = 8, make_plot = True):
        """
        Generate a dataset based on a thermal system's differential equations.

        Parameters:
        - file_name (str): Name of the output CSV file.
        - n_curves (int): Number of curves to generate.
        - make_plot (bool): Whether to plot the results.
        """
        # arrays for saving data 
        features = [] # t, P array 
        targets = [] # T array 
        for i in range(n_curves):
            power_addition = -0.2 + i*0.05 # TODO check math on this one 

            def _dTdt(T, t):
                # Differential equation definition 
                P_elec = self._test_power_input(power_addition, t)
                return (P_elec - self.h_base * self.s * (T - self.T_amb)) / (self.m * self.cp)

            time_points = np.linspace(1, self.N_t_points, self.N_t_points)

            # Solve the differential equation
            temperature_solution = odeint(_dTdt, self.T_initial, time_points)

            for k in range(len(time_points)):
                if k % self.save_rate == 0:
                    t = time_points[k]
                    power_t = self._test_power_input(power_addition, t)
                    features.append((t, power_t))
                    targets.append(temperature_solution[k])
            # Plot the results
            if make_plot:
                plt.plot(time_points, temperature_solution)

        df_features = pd.DataFrame(features, columns=['t', 'P'])
        df_targets = pd.DataFrame(targets, columns=['T'])

        print(f"Test Dataset of length {len(features)} has been generated.")

        df = pd.concat([df_features, df_targets], axis=1)
        df.to_csv(file_name, index=False)

        if make_plot:
            plt.xlabel("Time (s)")
            plt.ylabel("Temperature (K)")
            plt.title("Temperature Evolution Over Time with Stabilization")
            plt.show()


G = Generator(save_rate=10)
G.generate_test('dataset5_test.csv', make_plot=False)