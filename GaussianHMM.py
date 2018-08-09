class GaussianHMM:
    def __init__(self,no_state,no_cluster,StateProb,Gain,Mu,Sigma):
        self.no_state = no_state
        self.no_cluster = no_cluster
        self.StateProb = StateProb
        self.Gain = Gain
        self.Mu = Mu
        self.Sigma = Sigma
    def phi(self,o,state,cluster):
        return (1/np.sqrt(2*np.pi*self.Sigma[state][cluster]))*np.exp(-(((o-self.Mu[state][cluster])**2)/(2*self.Sigma[state][cluster])))
    def EmissProb(self,k,o):
        p = 0
        for i in range(self.no_cluster):
            p+=self.Gain[k][i]*self.phi(o,k,i)
        return p
    def alpha(self,O,T):
        A = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            A[0][k] = self.EmissProb(k,O[0])*self.StateProb[k]
        for t in range(1,T):
            for k in range(self.no_state):
                A[t][k]=self.EmissProb(k,O[t])*self.StateProb[k]*np.sum(A[t-1])
        return A[T-1]
    def beta(self,O,T,t):
        B = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            B[T-1][k] = 1
        time = T-2
        while(time>=t):
            for k in range(self.no_state):
                if np.sum([self.EmissProb(k,O[time+1]) for k in range(self.no_state)])==0:
                    raise Exception("Shit hit the ceiling",O[time+1])
                B[time][k]=np.sum([B[time+1][k]*self.StateProb[k]*self.EmissProb(k,O[time+1]) for k in range(self.no_state)])
            time=time-1
        if((np.sum(B[t]))==0):
            print("B is zero")
            print(B,t)
        return B[t]
    def getSequenceProb(self,O,T):
        Alpha = self.alpha(O,T)
        return np.sum(Alpha)
    def getStateProbability(self,O,T,t,k):
        Alpha = self.alpha(O,t+1)
        Beta = self.beta(O,T,t)
        #print(Alpha,Beta)
        for i in range(len(Alpha)):
            if(np.isnan(Alpha[i])):
                raise Exception("Major Error: check Alpha Function",i)
            if(np.isnan(Beta[i])):
                raise Exception("Major Error: check Beta Function",i,t)
        #print("Alpha and Beta:---")
        #print(Alpha,Beta)
        return Alpha[k]*Beta[k]/(np.sum([Alpha[i]*Beta[i] for i in range(self.no_state)]))
    def getAlphaMatrix(self,O,T):
        A = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            A[0][k] = self.EmissProb(k,O[0])*self.StateProb[k]
        for t in range(1,T):
            for k in range(self.no_state):
                A[t][k]=self.EmissProb(k,O[t])*self.StateProb[k]*np.sum(A[t-1])
        if(np.isnan(A[t][0])):
            print(A)
            raise Exception("Check Alpha Function")
        return A
    def getBetaMatrix(self,O,T):
        B = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            B[T-1][k] = 1
        time = T-2
        while(time>=0):
            for k in range(self.no_state):
                if np.sum([self.EmissProb(k,O[time+1]) for k in range(self.no_state)])==0:
                    raise Exception("Shit hit the ceiling",O[time+1])
                B[time][k]=np.sum([B[time+1][k]*self.StateProb[k]*self.EmissProb(k,O[time+1]) for k in range(self.no_state)])
            time=time-1
        if((np.sum(B))==0):
            print("B is zero")
            print(B)
        return B
    def E(self,O,T):
        gamma = [[0]*T for i in range(self.no_state)]
        Alpha = self.getAlphaMatrix(O,T)
        Beta = self.getBetaMatrix(O,T)
        for t in range(T):
            for k in range(self.no_state):
                if(np.isnan(Alpha[t][k]*Beta[t][k]/(np.sum([Alpha[t][i]*Beta[t][i] for i in range(self.no_state)])))):
                    raise Exception("Trace back from E",Alpha,Beta)
                gamma[k][t]=Alpha[t][k]*Beta[t][k]/(np.sum([Alpha[t][i]*Beta[t][i] for i in range(self.no_state)]))
        omega = [[[0]*T for k in range(self.no_cluster)] for i in range(self.no_state)]
        for i in range(self.no_state):
            for k in range(self.no_cluster):
                for t in range(T):
                    omega[i][k][t]=gamma[i][t]*self.Gain[i][k]*self.phi(O[t],i,k)/self.EmissProb(i,O[t])
                    if(np.isnan(omega[i][k][t])):
                        omega[i][k][t]=0
        return gamma,omega
    def M(self,gamma,omega,O,T):
        pi = [0]*self.no_state
        gain = [[0]*self.no_cluster for i in range(self.no_state)]
        mu = [[0]*self.no_cluster for i in range(self.no_state)]
        sigma = [[0]*self.no_cluster for i in range(self.no_state)]
        SumGamma = [np.sum(gamma[i]) for i in range(self.no_state)]
        #SumOmega = [[np.sum(omega[i][k]) for i in range(self.no_state)] for k in range(self.no_cluster)]
        
        for i in range(self.no_state):
            pi[i]=SumGamma[i]/T
            for k in range(self.no_cluster):
                gain[i][k]=np.sum([omega[i][k][t] for t in range(T)])/SumGamma[i]/SumGamma[i]
                mu[i][k]= np.sum([omega[i][k][t]*O[t] for t in range(T)])/SumGamma[i]
                sigma[i][k] = np.sum([omega[i][k][t]*((O[t]-mu[i][k])**2) for t in range(T)])/SumGamma[i]
                if((sigma[i][k]==0)|np.isnan(sigma[i][k])):
                    print(np.sum([omega[i][k][t]*((O[t]-mu[i][k])**2) for t in range(T)]),SumGamma[i])
                    #raise Exception("Sigma is Problem",i,k)
                    sigma[i][k]=0.001
                
        for i in range(len(gain)):
            su = np.sum(gain[i])
            for j in range(len(gain[0])):
                gain[i][j]/=su
        
        return pi,gain,mu,sigma
    def thrownoise(self,noise = 0.5):
        for i in range(len(self.Sigma)):
            for j in range(len(self.Sigma[0])):
                self.Sigma[i][j]+=noise
    def learn(self,O,T,no_iter=100,margin=0.001):
        self.thrownoise(noise=0.05)
        for iter in range(no_iter):
            gamma,omega = self.E(O,T)
            for i in range(len(gamma)):
                for j in range(len(gamma[0])):
                    if(np.isnan(gamma[i][j])):
                        raise Exception("Gamma is NAN",i,j)
            pi,gain,mu,sigma = self.M(gamma,omega,O,T)
            boolarray = abs(np.array(mu)-np.array(self.Mu))<margin
            boolarray2= abs(np.array(sigma)-np.array(self.Sigma))<margin
            if((boolarray.all())):
                #print("Converged")
                return
            
            self.StateProb = pi
            self.Gain = gain
            self.Mu = mu
            self.Sigma = sigma
            
        return 
    