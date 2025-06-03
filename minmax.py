class minmax:
    def __init__(self):
        with open("winequality-red.csv", 'r') as f:  # Автоматическое закрытие файла
            tdl = f.readlines()[1:]
        self.tdlsplit = [line.replace('"', '').strip().split(';') for line in tdl]
        self.maxlist = []
        self.minlist = []
        for i in range(len(self.tdlsplit[0])-1):
            column_values = []
            for j in self.tdlsplit:
                try:
                    val = float(j[i])
                    column_values.append(val)
                except (ValueError, IndexError):
                    continue
            if column_values:
                self.maxlist.append(max(column_values))
                self.minlist.append(min(column_values))
    
    def coefficient(self, num, cur):
        if self.maxlist[num] == self.minlist[num]:
            return 0.0  # Избегаем деления на ноль
        return (cur - self.minlist[num]) * 2 / (self.maxlist[num] - self.minlist[num]) - 1