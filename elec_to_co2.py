import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

dfe = pd.read_csv('electricity.csv')
dfc = pd.read_csv(
    'C:/Users/Ionap/OneDrive/Documents/Edinburgh/Dissertation/Carbon intensity.csv')

CO2 = (np.array(dfe['electricity'].tolist())*24/1000)*np.array(dfc['Carbon'].tolist())[9:]/1000
print(CO2)
df = pd.DataFrame({'CO2': CO2})
df.to_csv('CO2_solar_air_pump2.csv')