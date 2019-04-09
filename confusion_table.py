import numpy as np
import csv

hours = [1,2,4,12,24]

TP = 3
FP = 4
TN = 1
FN = 2

table = {}

for h in hours:

    ha1 = np.genfromtxt("alex/MooCall/2018-10-01_python/confusion_table_" + str(h) + "h_ha1.csv", delimiter=";")[1:]
    ha2 = np.genfromtxt("alex/MooCall/2018-10-01_python/confusion_table_" + str(h) + "h_ha2.csv", delimiter=";")[1:]

    tp1 = sum(ha1[:,TP])
    tp2 = sum(ha2[:,TP])

    tn1 = sum(ha1[:,TN])
    tn2 = sum(ha2[:,TN])

    fn1 = sum(ha1[:,FN])
    fn2 = sum(ha2[:,FN])

    fp1 = sum(ha1[:,FP])
    fp2 = sum(ha2[:,FP])

    table[h] = [tp1, tp2, fp1, fp2, tn1, tn2, fn1, fn2]

    print("Hour:", h, " alarms: ", tp1+tp2+fp1+fp2)

with open("ConfusionTable.csv", "w") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["Classification"] + [str(h) + "h" for h in hours])
    w.writerow(["True positives HA1"] + [table[h][0] for h in hours])
    w.writerow(["True positives HA2"] + [table[h][1] for h in hours])
    w.writerow(["False positives HA1"] + [table[h][2] for h in hours])
    w.writerow(["False positives HA2"] + [table[h][3] for h in hours])
    w.writerow(["True negatives HA1"] + [table[h][4] for h in hours])
    w.writerow(["True negatives HA2"] + [table[h][5] for h in hours])
    w.writerow(["False negatives HA1"] + [table[h][6] for h in hours])
    w.writerow(["False negatives HA2"] + [table[h][7] for h in hours])

