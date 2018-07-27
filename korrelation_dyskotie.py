import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

def parse_time(time):
    date_time = datetime.datetime.strptime(time, "%H:%M")
    return int(date_time.hour * 60 + date_time.minute)

with open("./Voss-Moocall-Geburtsverlauf_Korrelation_Dystokie_24-Jul-2018.csv") as fh:
    data = list(csv.reader(fh, delimiter=","))

labour_time0 = []
labour_time1 = []
labour_time2 = []
labour_time3 = []
labour_times = []
scores = []
for d in data[1:]:
    score = int(d[3])
    time = parse_time(d[5])
    scores.append(score)
    if score == 0:
        labour_time0.append(time)
    elif score == 1:
        labour_time1.append(time)
    elif score == 2:
        labour_time2.append(time)
    elif score == 3:
        labour_time3.append(time)
    labour_times.append(time)
    print(scores[-1], "vs", labour_times[-1], "(", d[3], "vs", d[5], ")")

scores = np.array(scores)
labour_times = np.array(labour_times)

labour_time0 = np.array(labour_time0)
labour_time1 = np.array(labour_time1)
labour_time2 = np.array(labour_time2)
labour_time3 = np.array(labour_time3)

print(labour_time0.mean())
print(labour_time1.mean())
print(labour_time2.mean())
print(labour_time3.mean())

x = [0,1,2,3]
y = [labour_time0.mean(), labour_time1.mean(), labour_time2.mean(), labour_time3.mean()]
e = [labour_time0.std(), labour_time1.std(), labour_time2.std(), labour_time3.std()]

plt.errorbar(x, y, yerr=e)
plt.plot(scores, labour_times, "bo")
plt.xticks([0,1,2,3])
plt.xlabel("Geburtsverlauf")
plt.ylabel("Geburtsdauer in Minuten")
plt.show()
