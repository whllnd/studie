import csv

with open("../2018-10-01_python_kuehe/2018-09-14_rohdaten_korr.csv") as fh:
    data = list(csv.reader(fh, delimiter=";"))

with open("../2018-10-01_python_kuehe/2018-09-14_rohdaten_korr_Kuehe.csv", "w") as fcows, open("../2018-10-01_python_faersen/2018-09-14_rohdaten_korr_Faersen.csv", "w") as fheif:
    cows = csv.writer(fcows, delimiter=";")
    cows.writerow(data[0])
    heif = csv.writer(fheif, delimiter=";")
    heif.writerow(data[0])

    for r in data[1:]:
        if r[1] == "Kuh":
            cows.writerow(r)
        elif r[1] == "Faerse":
            heif.writerow(r)
