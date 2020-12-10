from sklearn import datasets
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target

clusters = 3
samples, dimension = X.shape
iters = 100
def initial():
    random_data = np.random.randint(low=0, high=samples, size=clusters)
    mu = [X[i,:] for i in random_data]
    sigma = np.array([np.eye(dimension) for _ in range(clusters)])
    prior = np.ones(clusters)*1./clusters
    return mu, sigma, prior

def e_step(X, prior, mu, sigma):
    ws = np.zeros((clusters, samples))
    for i in range(clusters):
         distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
         ws[i,:] = prior[i]*distribution.pdf(X)
    normalized = ws/ws.sum(0)
    prior = normalized.mean(1)
    return normalized, prior

def m_step(X, E, mu, sigma):
    mu = np.dot(E,X)/E.sum(1)[:, None]
    for i in range(clusters):
        E_i = E[i,:]
        t = X-mu[i]
        sigma[i] = (E_i[:,None,None]*(t[:,:,None]*t[:,:,None].transpose(0,2,1))).sum(0)
    sigma /= E.sum(1)[:, None, None]
    return mu, sigma

def cal_log(X, prior, mu, sigma):
    l = 0
    for i in range(clusters):
        l += prior[i]*multivariate_normal(mu[i,:], sigma[i,:]).pdf(X)
    l = np.log(l).sum()
    return l

def predict(X, prior, mu, sigma):
    E, _ = e_step(X, prior, mu, sigma)
    return np.argmax(E, axis=0)

def train():
    mu, sigma, prior = initial()
    ll = []
    for iteration in range(iters):
        E, prior = e_step(X, prior, mu, sigma)
        mu,sigma = m_step(X, E, mu, sigma)
        ll.append(cal_log(X,prior,mu,sigma))
    return prior, mu, sigma, ll

prior, mu, sigma, ll = train()
print('prior', prior)
print('mu', mu)
print('sigma', sigma)
permuted_prediction = predict(X, prior, mu, sigma)
def eval(permuted_prediction):
    permutation = np.array([mode(iris.target[permuted_prediction == i]).mode.item() for i in range(clusters)])
    permuted_prediction = permutation[permuted_prediction]
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score
    print('error',mean_squared_error(iris.target, permuted_prediction))
    print('acc',accuracy_score(iris.target, permuted_prediction))
    return mean_squared_error(iris.target, permuted_prediction)

mse = eval(permuted_prediction)
#plt.plot(ll)
#plt.show()

def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

def calculate_mdl(n, mse, c):
    num_params = c *(c-1 + dimension + dimension*(dimension-1)/2.)
    return mse*(1+num_params*np.log(n)/n)

for c in [3,4,5,6,7]:
    clusters = c
    prior, mu, sigma, ll = train()
    permuted_prediction = predict(X, prior, mu, sigma)
    mse = eval(permuted_prediction)
    print('clusters',clusters,'mdl',calculate_mdl(samples, mse, clusters))
