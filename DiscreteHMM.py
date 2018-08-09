class HMM:
    def __init__(self,no_state,no_emiss,StateProb,EmissProb):
        self.no_state = no_state
        self.no_emiss = no_emiss
        self.StateProb = StateProb
        self.EmissProb = EmissProb
    def alpha(self,O,T):
        A = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            A[0][k] = self.EmissProb[k][O[0]]*self.StateProb[k]
        for t in range(1,T):
            for k in range(self.no_state):
                A[t][k]=self.EmissProb[k][O[t]]*self.StateProb[k]*np.sum(A[t-1])
        return A[T-1]
    def beta(self,O,T,t):
        
        #print(t,T)
        B = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            B[T-1][k] = 1
        time = T-2
        while(time>=t):
            for k in range(self.no_state):
                B[time][k]=np.sum([B[time+1][k]*self.StateProb[k]*self.EmissProb[k][O[time+1]] for k in range(self.no_state)])
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
            A[0][k] = self.EmissProb[k][O[0]]*self.StateProb[k]
        for t in range(1,T):
            for k in range(self.no_state):
                A[t][k]=self.EmissProb[k][O[t]]*self.StateProb[k]*np.sum(A[t-1])
        return A
    def getBetaMatrix(self,O,T):
        #print(t,T)
        B = [[0]*self.no_state for i in range(T)]
        for k in range(self.no_state):
            B[T-1][k] = 1
        time = T-2
        while(time>=0):
            for k in range(self.no_state):
                B[time][k]=np.sum([B[time+1][k]*self.StateProb[k]*self.EmissProb[k][O[time+1]] for k in range(self.no_state)])
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
                gamma[k][t]=Alpha[t][k]*Beta[t][k]/(np.sum([Alpha[t][i]*Beta[t][i] for i in range(self.no_state)]))
        return gamma
    def M(self,gamma,O,T):
        pi = [0]*self.no_state
        q = [[0]*self.no_emiss for i in range(self.no_state)]
        for i in range(self.no_state):
            pi[i] = np.sum(gamma[i])/T 
            for j in range(self.no_emiss):
                delta = [0]*T
                for t in range(T):
                    if(O[t]==j):
                        delta[t]=gamma[i][t]
                q[i][j] = np.sum(delta)/np.sum(gamma)
        for i in range(len(q)):
            su = np.sum(q[i])
            for j in range(len(q[0])):
                q[i][j]/=su
        return pi,q
    def thrownoise(self,noise=0.05):
        for i in range(len(self.EmissProb)):
            for j in range(len(self.EmissProb[0])):
                self.EmissProb[i][j]+=noise
        for i in range(len(self.EmissProb)):
            self.EmissProb[i]=self.EmissProb[i]/np.sum(self.EmissProb[i])
        
    def learn(self,O,T,no_iter=100,margin=0.001):
        self.thrownoise()
        for iter in range(no_iter):
            gamma = self.E(O,T)
            for i in range(len(gamma)):
                for j in range(len(gamma[0])):
                    if(np.isnan(gamma[i][j])):
                        raise Exception("Gamma is NAN",i,j)
            pi,q = self.M(gamma,O,T)
            boolarray = abs(np.array(q)-np.array(self.EmissProb))<margin
            if(boolarray.all()):
                return
            self.StateProb = pi
            self.EmissProb = q
        return 
    