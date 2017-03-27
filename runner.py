#this file executes NPL Simulations and fetches some parameters
# Usage: 'python runner.py [initial concentration] [equilibrium concentration] [Attempt frequency]'

import sys
import numpy as np

import nplSimusFinal

Z = nplSimusFinal.nplSimus()

Z.params[1] = float(sys.argv[4])
Z.modelSetup['y0'][0] = float(sys.argv[2])
Z.physicalProperties['cstar'] = float(sys.argv[3])

Z.solve()

#quick check
print Z.output['c']

#save the output
savestring = 'final.cstar'+sys.argv[3]+'kP'+str(Z.params[1])+'.npy'
np.save(savestring,Z.output)

#to load the saved output: np.load('my_file.npy').item()
