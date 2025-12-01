import numpy as np
x= np.loadtxt("global_representation.txt")
a = x[0]
np.savetxt("AGO1.txt",a)
print(1)