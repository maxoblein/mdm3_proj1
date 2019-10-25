import csv

with open('Default Measures.csv', 'r') as csvfile:
    measurements = csv.reader(csvfile)
    print(measurements)
    # Ball_width=[]
    # Heel_width=[]
    # for collumn in measurements:
        