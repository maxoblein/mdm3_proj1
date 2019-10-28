import matplotlib.pyplot as plt
from vrml_reader import find_coords
from vrml_reader import Visualise

    
inp = '000000.wrl'

data = find_coords(inp)

Visualise(data)
print(data)

if 