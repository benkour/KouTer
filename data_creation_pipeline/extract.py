import pandas as pd
import os
abs_path = os.path.dirname(os.path.abspath(__file__))

data = pd.read_excel(abs_path+'/solventSelectionTool_table.xlsx', skiprows=3)

print(data.columns)
health_hazard = data['Health Hazard']
boiling_point = data['Boiling Point (°C)']
melting_point = data['Melting Point (°C)']
dP = data['dP - Polarity']
dD = data['dD - Dispersion']
dH = data['dH - Hydrogen bonding']
names = data['Solvent Name']
# save data in a csv
df = pd.DataFrame({'Name': names, 'health_hazard': health_hazard,
                   'boiling_point': boiling_point, 'melting_point': melting_point, 'dD':dD, 'dH':dH, 'dP': dP})
df.to_csv(abs_path+'/environmental_properties.csv', index=False)
