#first Algo Metropolis-Hstings 
#Gibbs sampler is useless here because we suppose that delta and alpha are independant 
#They both follow a gaussian 

import numpy as np 
import scipy.stats as sp 
import matplotlib.pyplot as plt 

def heartbeats(sd, sigma, tau, nchain, after_drugs, before_drugs):
    PVC_count = after_drugs + before_drugs
    #2 parameters to estimate 
    chain = np.zeros((nchain+1, 2)) #alpha and delta 
    #init at 0,0 (told so in the data folder) --> p = 0.5 and theta = 0.5
    #rate of acceptance 
    acceptance = 0
    
    #density 
    def log_g(alpha, delta, y, t):
        #vector that indicates the value where the is no PVC's
        y_null = (y == 0) * 1
        n0 = y_null.sum()
        n1 = (1-y_null).sum()
        
        sum1 = -((alpha/sigma)**2 + (delta/tau)**2)/2
        
        sum2 = n1 * np.log(2 *np.exp(delta)/ (1+np.exp(delta))) + alpha * y.sum() - \
            np.log(1+np.exp(alpha)) * (t * (1-y_null)).sum()
            
        sum3 = (np.log(np.exp(delta) + (1+np.exp(alpha))**(-t)) * y_null).sum() - \
            -np.log(1+np.exp(delta)) * n0
        return sum1 +sum2+sum3
    
    for i in range(nchain):
        new_candidate = chain[i, :] + np.random.normal(size = 2, scale= sd)
        
        #proposal kernel symetric --> only the difference between the log-densities 
        ratio = np.exp(log_g(new_candidate[0], new_candidate[1], after_drugs,PVC_count) - \
            log_g(chain[i,0], chain[i, 1], after_drugs, PVC_count))
        #print(ratio)
        #MAJ 
        u = np.random.uniform()
        if u < ratio: 
            chain[i+1, : ] = new_candidate
            acceptance += 1
        else:
            chain[i+1, :] = chain[i, :]
        
        
        #using the inverse of logit to get p and theta ? 
        
    return chain, acceptance/nchain
        
#processing the data 
"""
N <-
12
t <-
c(11, 11, 17, 22, 9, 6, 5, 14, 9, 7, 22, 51)
x <-
c(6, 9, 17, 22, 7, 5, 5, 14, 9, 7, 9, 51)
y <-
c(5, 2, 0, 0, 2, 1, 0, 0, 0, 0, 13, 0)
"""
#number of individual 
N = 12
#PVC total count 
t = np.array([11, 11, 17, 22, 9, 6, 5, 14, 9, 7, 22, 51])

#before drugs 
x = np.array([6, 9, 17, 22, 7, 5, 5, 14, 9, 7, 9, 51])

#after drugs 
y = np.array([5, 2, 0, 0, 2, 1, 0, 0, 0, 0, 13, 0])

sigma = 1e-2
tau = 1e-2

chain, accep = heartbeats(.03, sigma, tau, 11000, y, t)
print(accep)
plt.plot(chain[1000:, 0])
plt.show()
plt.plot(chain[1000:, 1])
plt.show()

#proprièté des estimation 

alpha = chain[:, 0]
delta = chain[:, 1]

fig, ax = plt.subplot_mosaic([["A", "A"],
                              ["B", "B"]])
ax["A"].hist(alpha[1000:], bins= 'auto', label= r"Posteriori $\alpha$");
ax["A"].legend()
ax["B"].hist(delta[1000:], bins= 'auto', label= r"Posteriori $\delta$");
ax["B"].legend()
plt.show()

print(f"La valeur moyenne de alpha {alpha[1000:].mean()} et son ecart-type {alpha[1000:].std()}")
print(f"La valeur moyenne de alpha {delta[1000:].mean()} et son ecart-type {delta[1000:].std()}")
