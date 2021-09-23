'''
    this is a simulator for generating data
'''
import random

import pandas as pd

op = ["No Load", "Medium", "Risk"]

# row = pd.DataFrame([ [0,0,0,"No Load"] ], columns=['Voltage', 'Current', "Power", "Condition"])
# row.to_csv('cities.csv', index=False, mode='a')


# Five num summary
''' 
original not preserved
                                        Voltage ,   Power,     Current
        5 number summary (Tukey)
        min :                               0   ,   0         , 0
        1st Quartile (25%) :             79.1514,   0         , 0.0768
        2st Quartile (50% = median) :    213.114,   38.43     , 0.1769
        3st Quartile (75%) :             215.031,   101.43    , 0.3987
        max :                            304.34,    264.85    , 1.2273
        mean :                           160.6573,  68.065    , 0.3177
        std :                            93.1255,   68.941    , 0.3204
    
'''
voltage = list()
power = list()
current = list()
output_label = list()
# no op data
for i in range(100):
    V, I =0,0

    if i < 50:
        voltage.append(0)
        I = random.uniform(0, 1.2273)
        current.append(I)
    else:
        V = random.uniform(0, 304.34)
        voltage.append(V)
        current.append(0)
    power.append(V * I)
    output_label.append("No Load")

for i in range(600):
    # var = random.uniform(0, 75.5)
    # V = random.uniform(0, 304.34)
    # I = random.uniform(0, 1.2273)
    V, I =0,0
    while V in voltage and I in current:
        V = random.uniform(0, 304.34)
        I = random.uniform(0, 1.2273)

    P = V * I
    voltage.append(V)
    current.append(I)
    power.append(P)

    if P > 90:
        output_label.append("Risk")
    elif P > 25:
        output_label.append("Medium")
    elif P > 0:
        output_label.append("Low")
    else:
        output_label.append("No Load")

    # print(voltage[-1], current[-1], power[-1], output_label[-1])

# print(len(voltage))
# print(len(current))
# print(len(power))
# print(len(output_label))

assert len(voltage) == len(current) == len(power) == len(output_label), "lists length does not match"

csv_dict = {
    "Current": current,
    "Voltage": voltage,
    "Power": power,
    "Output Condition" : output_label
}

row = pd.DataFrame(csv_dict)
row.to_csv('power_monitor_data.csv', index=False, mode='a')


# print(voltage)
# print(current)
# print(power)
# for i,output in enumerate(output_label):
#     print(current[i], voltage[i], power[i], output)

# for no load
# for i in range(400):
#     # cities = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida']], columns=['City', 'State'])
#     row = pd.DataFrame([[0,0,0,"No Load"]])
#     row.to_csv('cities.csv', index=False, mode='a')
#
# for i in range(400):
#     # cities = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida']], columns=['City', 'State'])
#     row = pd.DataFrame([[0,0,0,"Medium"]])
#     row.to_csv('cities.csv', index=False, mode='a')
#
# for i in range(400):
#     # cities = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida']], columns=['City', 'State'])
#     row = pd.DataFrame([[0,0,0,"Risk"]])
#     row.to_csv('cities.csv', index=False, mode='a')