from bayesian_blocks import BBlocks
import time
import numpy as np

bblocks = BBlocks()
# x=[0,0,0,0,0,0,0,0,2,0,1,0,0,1,0,1,3,0,3,0,3,3,0,1,1,3,2,2,1,2,0,1,2,1,0,2,0,1,1,2,3,3,3,1,0,1,2,2]
with open('times.txt', 'r') as file:
    # Legge tutte le righe e converte in array di numeri interi
    t = np.array([int(line.strip()) for line in file.readlines()])
with open('curve.txt', 'r') as file:
    x = np.array([int(line.strip()) for line in file.readlines()])

bblocks.bayesian_blocks(t,x,gamma=0.75)
print(bblocks.get_data_in())
print(bblocks.get_data_out())
bblocks.save_to_json("serialize_py.json")
bblocks.load_from_json("serialize_py.json")
print(bblocks.get_data_in())
print(bblocks.get_data_out())

print(".")
time.sleep(0.1)
print(".")
time.sleep(0.1)
print(".")
time.sleep(0.1)
bblocks = BBlocks()
bblocks.load_from_json("serialize.json")
print(bblocks.get_data_in())
print(bblocks.get_data_out())