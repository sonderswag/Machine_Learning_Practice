from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1

    # print a.shape , b.shape, b.T.shape

    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 4        # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1)) # generates random points for data evalutation 
print X

y = f(X) + s*np.random.randn(N) # these are the true values with a little bit of noise 
print y

K = kernel(X, X) # this is creating the kernel of the know data that we have 
# print K
# line 1 in pseudo code 
L = np.linalg.cholesky(K + s*np.eye(N)) # doing L = cholesky(K+sigma^2*I) this basically diagonalises it 
# print 
# print L 

# points we're going to make predictions : in order to plot the function.
Xtest = np.linspace(-5, 5, n).reshape(-1,1) # this is the grid of X* needed to plot the function 
# print Xtest


# compute the mean at our test points. 
Lk = np.linalg.solve(L, kernel(X, Xtest)) #l
# print "LK", Lk
mu = np.dot(Lk.T, np.linalg.solve(L, y))   # line 2 

#different from sudo code: in here it does the following 
# mu = [L\K_*].T dot [L\y]

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest) # this is the K_* that is going to append the K - covarience matrix 
print K_.shape
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
print L.shape
f_prior = np.dot(L, np.random.normal(size=(n,10)))
# print f_prior.shape
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()