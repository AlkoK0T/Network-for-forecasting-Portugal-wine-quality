class minmax:
    def __init__(self):
        self.tdf = open("winequality-red.csv", 'r')
        self.tdl =self.tdf.readlines()[1:]
        self.tdlsplit=[line.split(';') for line in self.tdl]
        self.maxlist = []
        self.minlist = []
        for i in range(len(self.tdlsplit[0])-1):
            self.maxlist.append(max([float(j[i]) for j in self.tdlsplit]))
            self.minlist.append(min([float(j[i]) for j in self.tdlsplit]))
    def coefficient(self,num,cur):
        return (cur-self.minlist[num])*2/(self.maxlist[num] - self.minlist[num])+(-1)
    
c=minmax()
