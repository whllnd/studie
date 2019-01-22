import csv

with open("./2018-11-02_r_output_kuehe.txt") as fh:
    data = fh.readlines()
    data = [x.strip() for x in data]

def check(hour, alarm):
    if hour is None or alarm is None:
        raise "Lol"

sens = {1:["","",""], 2:["","",""], 4:["","",""], 12:["","",""], 24:["","",""]} # [0] = ha1 und ha2, [1] = ha1, [2] = ha2
spec = {1:["","",""], 2:["","",""], 4:["","",""], 12:["","",""], 24:["","",""]}
prev = {1:["","",""], 2:["","",""], 4:["","",""], 12:["","",""], 24:["","",""]}
ppv  = {1:["","",""], 2:["","",""], 4:["","",""], 12:["","",""], 24:["","",""]}
npv  = {1:["","",""], 2:["","",""], 4:["","",""], 12:["","",""], 24:["","",""]}

hours = [1,2,4,12,24]
hour = None
alarm = None # 0 = beide, 1 = ha1, 2 = ha2

sprev = "True prevalence                        "
ssens = "Sensitivity                            "
sspec = "Specificity                            "
sppv  = "Positive predictive value              "
snpv  = "Negative predictive value              "

for r in data:
    if "Zeitraum:" in r:

        # Stunde
        hour = int(r.split("Zeitraum: ")[1].split(" ")[0])

        # Alarmtyp
        if "_ha1" in r:
            alarm = 1
        elif "_ha2" in r:
            alarm = 2
        else:
            alarm = 0

    if hour not in hours:
        continue

    if sprev in r:
        check(hour, alarm)
        print(r.split(sprev)[1], alarm)
        prev[hour][alarm] = r.split(sprev)[1]
    if ssens in r:
        check(hour, alarm)
        sens[hour][alarm] = r.split(ssens)[1]
    if sspec in r:
        check(hour, alarm)
        spec[hour][alarm] = r.split(sspec)[1]
    if sppv in r:
        check(hour, alarm)
        ppv[hour][alarm] = r.split(sppv)[1]
    if snpv in r:
        check(hour, alarm)
        npv[hour][alarm] = r.split(snpv)[1]

with open("Auswertung_Kuehe.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["", "1 Hour", "2 Hours", "4 Hours", "12 Hours", "24 Hours"])
    f.writerow(["Sensitivity"])
    f.writerow(["HA1 & HA2"] + [sens[hour][0] for hour in hours])
    f.writerow(["HA1"] + [sens[hour][1] for hour in hours])
    f.writerow(["HA2"] + [sens[hour][2] for hour in hours])
    f.writerow(["Specificity"])
    f.writerow(["HA1 & HA2"] + [spec[hour][0] for hour in hours])
    f.writerow(["HA1"] + [spec[hour][1] for hour in hours])
    f.writerow(["HA2"] + [spec[hour][2] for hour in hours])
    f.writerow(["True prevalence"])
    f.writerow(["HA1 & HA2"] + [prev[hour][0] for hour in hours])
    f.writerow(["HA1"] + [prev[hour][1] for hour in hours])
    f.writerow(["HA2"] + [prev[hour][2] for hour in hours])
    f.writerow(["PPV"])
    f.writerow(["HA1 & HA2"] + [ppv[hour][0] for hour in hours])
    f.writerow(["HA1"] + [ppv[hour][1] for hour in hours])
    f.writerow(["HA2"] + [ppv[hour][2] for hour in hours])
    f.writerow(["NPV"])
    f.writerow(["HA1 & HA2"] + [npv[hour][0] for hour in hours])
    f.writerow(["HA1"] + [npv[hour][1] for hour in hours])
    f.writerow(["HA2"] + [npv[hour][2] for hour in hours])
