import numpy as np
import scipy.special
from minmax import minmax
from tkinter import *
from tkinter import ttk
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
        for i in range(len(self.nodeslist)-1):
            nodes.append(np.dot(self.weight[i],cur[i]))
            cur.append(self.act_func(nodes[-1]))
        errors=[targ-cur[-1]] 
        for i in range(len(self.weight)-1, 0,-1):
            errors.append(np.dot(self.weight[i].T,errors[-1]))
        errors=errors[::-1]
        cur=cur[1:]
        for i in range(len(errors)-1, 0,-1):
            eri=errors[i]
            curi=cur[i]
            test=errors[i]*cur[i]*(1.0-cur[i])
            self.weight[i]+= self.lrate*np.dot((errors[i]*cur[i]*(1.0-cur[i])),np.transpose(cur[i-1]))   
        pass
    def epoch(self, file, epochs):
                # region ttk
        root = Tk()
        root.title("Train indicator")
        root.geometry("300x150") 
        epoch_var = IntVar()
        trainset_var = IntVar()
        efficienty=IntVar()
        epochbar =  ttk.Progressbar(orient="horizontal", maximum=epochs, variable=epoch_var)
        epochbar.grid(row=1, column=1, columnspan=5)
        epochtext = ttk.Label(text="Эпоха")
        epochtext.grid(row=1, column=6)
        epochvar = ttk.Label(textvariable=epoch_var)
        epochvar.grid(row=1, column=7)
        trainsetbar =  ttk.Progressbar(orient="horizontal", maximum=len(file), variable=trainset_var)
        trainsetbar.grid(row=2, column=1, columnspan=5)
        trainsettext = ttk.Label(text="Тренировочный")
        trainsettext.grid(row=2, column=6)
        trainsetvar = ttk.Label(textvariable=trainset_var)
        trainsetvar.grid(row=2, column=7) 
        effitext = ttk.Label(text="Эффективность")
        effitext.grid(row=3, column=1)
        effivar = ttk.Label(textvariable=efficienty)
        effivar.grid(row=3, column=2) 
        counter=[]
        #endregion
        minm=minmax()
        for i in range(epochs):
            for rec in file:
                all_val=[j for j in rec.split(';')]
                targets = np.zeros(11)
                targets[int(all_val[-1])+1]=1
                chngevar=[minm.coefficient(j,float(all_val[j])) for j in range(len(all_val)-1) ]
                inputs = (np.asarray(chngevar,dtype=float))
                self.train(inputs,targets)
                trainsetbar.step(1)
                root.update()
                pass
            epochbar.step(1)
            root.update()
        root.mainloop()
    def query(self,input_list):
        inp = np.array(input_list,ndmin=2).T
        cur=[inp]
        nodes=[]
        for i in range(len(self.nodeslist)-1):
            nodes.append(np.dot(self.weight[i],cur[i]))
            cur.append(self.act_func(nodes[-1]))     
        return cur[-1]   
