import numpy as np
import scipy.special


class network:
    def __init__(self, nodeslist, lrate):
        self.nodeslist = nodeslist
        self.lrate=lrate
        self.act_func = lambda x:scipy.special.expit(x)
        self.weight = []
        for i in range(1,len(nodeslist)):
            self.weight.append(np.random.normal(0.0,pow(nodeslist[i],-0.5), (nodeslist[i],nodeslist[i-1])))
        pass
    def train(self,input_list,target_list):
        inp = np.array(input_list,ndmin=2).T
        targ = np.array(target_list,ndmin=2).T
        cur=[inp]
        nodes=[]
        nodesact=[]
        for i in range(len(self.nodeslist)-1):
            nodes.append(np.dot(self.weight[i],cur[i]))
            cur.append(self.act_func(nodes[-1]))
        errors=[np.transpose(targ-cur[-1])] 
        for i in range(len(self.weight)-1, 0,-1):
            errors.append(np.dot(self.weight[i].T,errors[-1]))
        for i in range(len(self.weight)-1, 0,-1):
            self.weight[i]+= self.lrate*np.dot((errors[i]*cur[i]*(1.0-cur[i])),np.transpose(cur[i-1]))   
        pass


hiddennodes = [3,10,20,10,1]
c=network(hiddennodes,0.1)
c.train([1.1,2.2,3.3],[1.1,3.3,2.2])