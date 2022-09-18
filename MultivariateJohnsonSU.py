#Адаптирован код, размещенный здесь: https://github.com/chrsbats/connorav/blob/master/README.md

from scipy.stats import johnsonsu, norm
from scipy.optimize import minimize_scalar
from scipy import integrate
import pandas as pd
import numpy

from scipy.linalg import eigh, cholesky
from scipy.stats import norm, johnsonsu,spearmanr,skew, kurtosis

NORMAL_CUTOFF = 0.01

class MSSKDistribution(object):
  
    def __init__(self, mean=None, std=None, skew=None, kurt=None):
        if isinstance(mean,numpy.ndarray) or mean != None:
            self.fit(mean,std,skew,kurt)

    def fit(self, mean, std=None, skew=None, kurt=None):
        if std == None:
            #Array or tuple format.
            self.m = mean[0]
            self.s = mean[1]
            self.skew = mean[2]
            self.kurt = mean[3]
        else:
            self.m = mean
            self.s = std
            self.skew = skew
            self.kurt = kurt

        if abs(self.skew) < NORMAL_CUTOFF and abs(self.kurt) < NORMAL_CUTOFF:  
            #It is hard to solve the johnson su curve when it is very close to normality, so just use a normal curve instead.
            self.dist = norm(loc=self.m,scale=self.s)
            self.skew = 0.0
            self.kurt = 0.0

        else:
            a,b,loc,scale = self._johnsonsu_param(self.m,self.s,self.skew,self.kurt)
            self.dist = johnsonsu(a,b,loc=loc,scale=scale)

    def _optimize_w(self,w1,w2,b1,b2):
        def m_w(w):
            m = -2.0 + numpy.sqrt( 4.0 + 2.0 * ( w ** 2.0 - (b2 + 3.0) / (w ** 2.0 + 2.0 * w + 3.0)))
            return m

        def f_w(w):
            m = m_w(w)
            fw = (w - 1.0 - m) * ( w + 2.0 + 0.5 * m) ** 2.0
            return (fw - b1) ** 2.0

        
        if abs(w1 - w2) > 0.1e-6:
            solution = minimize_scalar(f_w, method='bounded',bounds=(w1,w2))
            w = solution['x']
        else:
            if w1 < 1.0001:
                w = 1.0001
            else:
                w = w1

        m = m_w(w)    

        return w, m


    def _johnsonsu_param(self,mean,std_dev,skew,kurt):
        #"An algorithm to determine the parameters of SU-curves in the johnson system of probabillity distributions by moment matching", HJH Tuenter, 2001
        
        #First convert the parameters into the moments used by Tuenter's alg. 
        u2 = std_dev ** 2.0
        u3 = skew * std_dev ** 3.0
        u4 = (kurt + 3.0) * std_dev ** 4.0
        b1 = u3 ** 2.0 / u2 ** 3.0
        b2 = kurt + 3.0

        w2 = numpy.sqrt((-1.0 + numpy.sqrt(2.0 * (b2 -1.0))))
        big_d = (3.0 + b2) * (16.0 * b2 * b2 + 87.0 * b2 + 171.0) / 27
        d = -1.0 + (7.0 + 2.0 * b2 + 2.0 * numpy.sqrt(big_d)) ** (1.0 / 3.0) - (2.0 * numpy.sqrt(big_d) - 7.0 - 2.0 * b2) ** (1.0 / 3.0)
        w1 = (-1.0 + numpy.sqrt(d) + numpy.sqrt( 4 / numpy.sqrt(d) - d - 3.0)) / 2.0
        if (w1 - 1.0) * ((w1 + 2.0) ** 2.0) < b1:
            #no curve will fit
            raise Exception("Invalid parameters, no curve will fit")

        w, mw = self._optimize_w(w1,w2,b1,b2)

        z = ((w + 1.0) / (2.0 * w )) * ( ((w - 1.0) / mw) - 1.0) 
        if z < 0.0:
            z = 0.0
        omega = -1.0 * numpy.sign(u3) * numpy.arcsinh(numpy.sqrt(z))
        
        a = omega / numpy.sqrt(numpy.log(w))
        b = 1.0 / numpy.sqrt(numpy.log(w))

        z =  w - 1.0 - mw
        if z < 0.0:
            z = 0.0
        loc = mean - numpy.sign(u3) * (std_dev / (w -1.0)) * numpy.sqrt(z)
        
        scale = std_dev / (w - 1.0) * numpy.sqrt( (2.0 * mw) / ( w + 1.0))

        return a,b,loc,scale

    def _cvar(self,upper=0.05,samples=64,lower=0.00001):
        interval = (upper - lower) / float(samples)
        ppfs = self.dist.ppf(numpy.arange(lower, upper+interval, interval))
        result = integrate.romb(ppfs, dx=interval)
        return result
    
    #Visible scipy methods for distribution objects. 
    #Note that scipy uses some funky metaprogramming.  It's easier to do this than to inherit from rv_continuous.
    def rvs(self,x=None):
        return self.dist.rvs(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def logcdf(self, x):
        return self.dist.logcdf(x)

    def sf(self, x):
        return self.dist.sf(x)

    def logsf(self, x):
        return self.dist.logsf(x)

    def ppf(self, x):
        return self.dist.ppf(x)

    def isf(self, x):
        return self.dist.isf(x)

    def mean(self):
        return self.dist.mean()

    def median(self):
        return self.dist.median()
    
    def std(self):
        return self.dist.std()

    def var(self):
        return self.dist.var()

    def stats(self):
        return self.m, self.s, self.skew, self.kurt
    
    import numpy 


class MultiJSU(object):
    import pandas as pd
    def __init__(self,data,method='cholesky',normal=False):
        
        
        stats_init=pd.DataFrame({'Mean':data.mean(),
                      'Std':data.std(),
                      'Skewness':skew(data),
                      'Kurtosis':kurtosis(data)})
        if normal==True:
            stats_init['Skewness']=0
            stats_init['Kurtosis']=0
        self.moments=numpy.asarray(stats_init)
        self.correlations=data.corr()      
        self.dimensions = self.correlations.shape[0]
        self.distributions = [MSSKDistribution(self.moments[i]) for i in range(data.shape[1])]

        
        
    def JSU_fit(self,data):
        a,b,loc,scale=johnsonsu.fit(data)
        return johnsonsu(a,b,loc,scale)    

    def generate(self,num_samples,method='cholesky'):
        rv = self._uniform_correlated(self.dimensions,self.correlations,num_samples,method)
        rv = rv.tolist()
        for d in range(self.dimensions):
            rv[d] = self.distributions[d].ppf(rv[d])
        self.rv = numpy.array(rv)
        return self.rv
        

    def _normal_correlated(self,dimensions,correlations,num_samples,method='cholesky'):

        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        x = norm.rvs(size=(dimensions, num_samples))

        # We need a matrix `c` for  which `c*c^T = r`.  We can use, for example,
        # the Cholesky decomposition, or the we can construct `c` from the
        # eigenvectors and eigenvalues.
        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(correlations, lower=True)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correlations)
            # Construct c, so c*c^T = r.
            c = numpy.dot(evecs, np.diag(np.sqrt(evals)))

        # Convert the data to correlated random variables. 
        y = numpy.dot(c, x)
        return y

    def _uniform_correlated(self,dimensions,correlations,num_samples,method='cholesky'):
        #print(correlations)
        #корректируем корреляционную матрицу, чтобы после генерации случайных чисел из 
        #нормального распределения корреляции были ближе к эмпирическим
        adj_corr=2*numpy.sin((numpy.pi/6)*correlations)
        #print(adj_corr)
        normal_samples = self._normal_correlated(dimensions,adj_corr,num_samples,method) 
        x = norm.cdf(normal_samples)
        #print(numpy.corrcoef(x))
        return x