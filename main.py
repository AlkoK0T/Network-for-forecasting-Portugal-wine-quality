from network import network
from minmax import minmax
import numpy as np

hiddennodes = [11,100,11]
tdf = open("winequality-red.csv", 'r')
tdl = tdf.readlines()
tdl = tdl[1:]
tdf.close()
networkred=network(hiddennodes,0.1)
networkred.epoch(tdl,1000)
#region result
tedf = open("winequality-red.csv", 'r')
tedl = tedf.readlines()
tedl = tedl[1:]
tedf.close()
pr=0
minm = minmax()
scorecard = []
predicted_labels = []  # Создаем список для хранения предсказанных меток

for rec in tedl:
    all_val=[j.replace('"', '').strip() for j in rec.split(';')]
    try:
        true_label = int(float(all_val[-1]))
        chngevar=[minm.coefficient(j,float(all_val[j])) for j in range(len(all_val)-1) ]
        inputs = (np.asarray(chngevar,dtype=float))
        out=networkred.query(inputs)
        predicted_label = np.argmax(out) - 1  # Преобразование one-hot обратно в метку
        predicted_labels.append(predicted_label)  # Сохраняем предсказанную метку

        if predicted_label == true_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    except (ValueError, IndexError) as e:
        print(f"Ошибка обработки записи: {rec}, {e}")
        continue

scorecard_array = np.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)

# Выводим уникальные предсказанные метки
print("Уникальные предсказанные метки:", set(predicted_labels))
print("Распределение меток:")
for label in sorted(set(predicted_labels)):
    count = predicted_labels.count(label)
    print(f"Метка {label}: {count} раз ({count/len(predicted_labels)*100:.1f}%)")
#endregion