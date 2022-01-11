import matplotlib.pyplot as plt
import numpy as np
E_all = 1   # assume the initial concentration of E is 1 uM, and E-all = E + ES because of mass action 
k1 = 100
k2 = 600
k3 = 150
S = np.arange(0, 300, 0.1)
 
v = k3*E_all*S / (S+(k2+k3) / k1)  # assume the rate of change of the ES is 0. then we can calculate this formular
 
plt.title("S-v")
plt.plot(S, v)
 
plt.show()