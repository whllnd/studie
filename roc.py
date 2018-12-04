import numpy as np
import matplotlib.pyplot as plt
import csv

with open("roc.csv") as fh:
	data = list(csv.reader(fh, delimiter=";"))[2:]

# Offsets to make live a bit easier
HA12, HA1, HA2 = 2, 9, 16
SE = 0
SP = 1
SEM = 2
SEP = 3
SPM = 4
SPP = 5

sens_ha12 = np.array([0] + [float(data[i][HA12+SE]) for i in range(len(data))] + [1])
spec_ha12 = 1 - np.array([1] + [float(data[i][HA12+SP]) for i in range(len(data))] + [0])

sens_ha1 = np.array([0] + [float(data[i][HA1+SE]) for i in range(len(data))] + [1])
spec_ha1 = 1 - np.array([1] + [float(data[i][HA1+SP]) for i in range(len(data))] + [0])

sens_ha2 = np.array([0] + [float(data[i][HA2+SE]) for i in range(len(data))] + [1])
spec_ha2 = 1 - np.array([1] + [float(data[i][HA2+SP]) for i in range(len(data))] + [0])

#for l in [[sens_ha12, spec_ha12, "HA1h+HA2h"], [sens_ha1, spec_ha1, "HA1h"], [sens_ha2, spec_ha2, "HA2h"]]:
#	plt.plot(l[1], l[0], "x-")
#	plt.text(l[1][1], l[0][1], "2h")
#	plt.text(l[1][3], l[0][3], "4h")
#	plt.text(l[1][5], l[0][5], "6h")
#	plt.text(l[1][11], l[0][11], "12h")
#	plt.title("ROC " + l[2] + " alarms")
#	plt.xlabel("1 - specificity")
#	plt.ylabel("sensitivity")
#	plt.axis([0,1,0,1])
#	plt.show()

with open("ROC_Sens_Spec.csv", "w") as fh:
	w = csv.writer(fh, delimiter=";")
	w.writerow(["Zeitraum VP", "Sensitivitaet HA1h+HA2h", "1 - Spezifitaet HA1h+HA2h", "Sensitivitaet HA1h", "1 - Spezifitaet HA1h", "Sensitivitaet HA2h", "1 - Spezifitaet HA2h"])
	w.writerow(["", 0, 0, 0, 0, 0, 0])
	for i,v in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24]):
		w.writerow([str(v) + "h", sens_ha12[i], spec_ha12[i], sens_ha1[i], spec_ha1[i], sens_ha2[i], spec_ha2[i]])
	w.writerow(["", 1, 1, 1, 1, 1, 1])

