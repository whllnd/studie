library(readr)
library(data.table)
daten_moocall <- read_delim("Rohdaten/Rohdaten_10-Sep-2018_Faersen.csv",
                                   ";", escape_double = FALSE, trim_ws = TRUE)

print_tables <- function(confusion_table) {
  table = epitools::epitable(c(sum(confusion_table$RP),sum(confusion_table$FP), sum(confusion_table$FN), sum(confusion_table$RN)))
  test.table <- epiR::epi.tests(table)
  print(test.table)
  matrix <- matrix(c(sum(confusion_table$RP), sum(confusion_table$FN), sum(confusion_table$FP),  sum(confusion_table$RN)), nrow=2)
  print(caret::confusionMatrix(as.table(matrix)))
  print(paste("Youden-Score:", mean(confusion_table$Sensitivitaet, na.rm = T) + mean(confusion_table$Spezifitaet, na.rm = T) -1))
}

setDT(daten_moocall)
daten_moocall <- daten_moocall[`Gekalbt (ja/nein)` == "ja"]
daten_moocall[, Zeit_bis_Umstallung := difftime(Umstallzeit, Alarmzeit, units = "hours")]
daten_moocall[, Sensorzeit := difftime(Umstallzeit, SonT, units = "hours")]
daten_moocall[, Studienzeit := max(Sensorzeit, na.rm = T), by="ID"]
daten_moocall[is.na(Zeit_bis_Umstallung), Zeit_bis_Umstallung := 99999]
daten_moocall <- daten_moocall[Zeit_bis_Umstallung >=0]
daten_moocall <- daten_moocall[!ID %in% c(1899, 3909, 2641, 1304)]
uniqueN(daten_moocall$ID)

uniqueN(daten_moocall[`Alarm (ja/nein)` == "ja"]$ID)/uniqueN(daten_moocall$ID)
#c(1,2,3,4,6)
for (i in 1:12) {
  print(paste("Zeitfenster:", i, "Sensitivität:", uniqueN(daten_moocall[Zeit_bis_Umstallung <= i]$ID)/uniqueN(daten_moocall$ID)))
}

#### Nur HA 2
for (i in 1:12) {
  print(paste("Zeitfenster:", i, "Sensitivität:", uniqueN(daten_moocall[Zeit_bis_Umstallung <= i & `Alarmtyp (HA1/HA2)` == "HA2h"]$ID)/uniqueN(daten_moocall$ID)))
}

#### Nur HA 1
for (i in 1:12) {
  print(paste("Zeitfenster:", i, "Sensitivität:", uniqueN(daten_moocall[Zeit_bis_Umstallung <= i & `Alarmtyp (HA1/HA2)` == "HA1h"]$ID)/uniqueN(daten_moocall$ID)))
}

#`Alarmtyp (HA1/HA2)` == "HA2h"
daten_moocall <- daten_moocall[order(Alarmzeit)]
daten_moocall[order(Alarmzeit), Abstand_vorheriger_Alarm := difftime(Alarmzeit, shift(Alarmzeit, n = 1L, type = "lag"), units = "hour"), by = "ID"]
daten_moocall[`Alarmtyp (HA1/HA2)` == "HA2h", Abstand_HA2_davor := difftime(Alarmzeit, shift(Alarmzeit, n = 1L, type = "lag"), units = "hour"), by = "ID"]
table(daten_moocall$Abstand_vorheriger_Alarm)
table(daten_moocall$Abstand_HA2_davor)

daten_moocall[`Alarmtyp (HA1/HA2)` == "HA2h" & Abstand_vorheriger_Alarm>2]

# PPV
for (i in 1:12) {
  Anzahl_Kuehe_mit_korrektem_Alarm = uniqueN(daten_moocall[Zeit_bis_Umstallung <= i]$ID)
  Anzahl_Fehlalarme = nrow(daten_moocall[Zeit_bis_Umstallung > i][`Alarm (ja/nein)` == "ja"])
  print(paste("Zeitfenster:", i, "PPV:", Anzahl_Kuehe_mit_korrektem_Alarm/(Anzahl_Fehlalarme + Anzahl_Kuehe_mit_korrektem_Alarm)))
}

for (i in 1:12) {
  Anzahl_Kuehe_mit_korrektem_Alarm = uniqueN(daten_moocall[Zeit_bis_Umstallung <= i & `Alarmtyp (HA1/HA2)` == "HA2h"]$ID)
  Anzahl_Fehlalarme = nrow(daten_moocall[Zeit_bis_Umstallung > i][`Alarmtyp (HA1/HA2)` == "HA2h"])
  print(paste("Zeitfenster:", i, "PPV:", Anzahl_Kuehe_mit_korrektem_Alarm/(Anzahl_Fehlalarme + Anzahl_Kuehe_mit_korrektem_Alarm)))
}

daten_moocall[Zeit_bis_Umstallung > 0][`Alarm (ja/nein)` == "ja"][,.N, by="ID"][order(N)]
daten_moocall[Zeit_bis_Umstallung > 0][`Alarmtyp (HA1/HA2)` == "HA2h"][,.N, by="ID"][order(N)]


for (i in 3) {

daten_moocall[,Zeitintervall := as.integer(floor(Zeit_bis_Umstallung/i))]

zeitintervalle = daten_moocall[, .(HA2 = ifelse("HA2h" %in% `Alarmtyp (HA1/HA2)`, 1,0),
                      HA1 = ifelse("HA1h" %in% `Alarmtyp (HA1/HA2)`, 1,0),
                      Studienintervalle = as.integer(ceiling(max(Studienzeit)/i)),
                      Fehler = sum(as.integer(2  == `Event Score`), na.rm = T) + sum(as.integer(3  == `Event Score`), na.rm = T) + sum(as.integer(4  == `Event Score`), na.rm = T)
                      ) ,by=c("ID","Zeitintervall")]

#nrow(zeitintervalle[Zeitintervall>0][HA1 == 1 | HA2 == 1])
#zeitintervalle[Zeitintervall<10000 & Zeitintervall<Studienintervalle]

confusion_table <- zeitintervalle[, .(RN=max(Studienintervalle)-(.N-1), #-as.integer(sum(Fehler))
                              FN = as.numeric(min(Zeitintervall)!=0),
                              RP = sum(as.numeric(Zeitintervall==0)),
                              FP = sum(as.numeric(Zeitintervall<33000 & Zeitintervall>0))
                              ) , by="ID"]

table = epitools::epitable(c(sum(confusion_table$RP),sum(confusion_table$FP), sum(confusion_table$FN), sum(confusion_table$RN)))
print(paste("Zeitintervall", i))
print(epiR::epi.tests(table))

}

confusion_table[, `:=` (
  Sensitivitaet = RP/(RP+FN),
  Spezifitaet = RN/(RN+FP),
  PPV = RP/(RP+FP),
  NPV = RN/(RN+FN)
)]

confusion_table[Spezifitaet<0, Spezifitaet:= NA]
confusion_table[NPV<0, NPV:= NA]

#setDT(X2018_09_14_Rohdaten_korr)
#kuh_faerse <- X2018_09_14_Rohdaten_korr[, .(Kuh = unique(`Kuh/Faerse`)), by=ID]

a = c("", "_ha1", "_ha2")
b = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24)

out = ""

sink("2018-11-02_r_output_faersen.txt")

for (Stunde in b) {
  for (Alarmtyp in a) {
    #file_name = paste0("3. Beratung/2018-10-01_python/confusion_table_", Stunde, "h", Alarmtyp, ".csv")
    file_name = paste0("./2018-10-01_python_faersen/confusion_table_", Stunde, "h", Alarmtyp, ".csv")
    print(file_name)
    confusion_table <- fread(file = file_name)
    print(paste0("Zeitraum: ", Stunde, " Alarm: ", Alarmtyp))
    out = c(out,print_tables(confusion_table = confusion_table))
  }
}

sink()
print("bontastisch")
#confusion_table <- merge(confusion_table, kuh_faerse, by="ID", all.x = T)
#confusion_table <- confusion_table[Kuh=="Faerse"]

#mean(confusion_table$Sensitivitaet, na.rm = T)
#mean(confusion_table$Spezifitaet, na.rm = T)
#mean(confusion_table$PPV, na.rm = T)
#mean(confusion_table$NPV, na.rm = T)

#(mean(confusion_table$Sensitivitaet, na.rm = T) + mean(confusion_table$Spezifitaet, na.rm = T))/2

#print_tables <- function(confusion_table) {
#  table = epitools::epitable(c(sum(confusion_table$RP),sum(confusion_table$FP), sum(confusion_table$FN), sum(confusion_table$RN)))
#  test.table <- epiR::epi.tests(table)
#  print(test.table)
#  matrix <- matrix(c(sum(confusion_table$RP), sum(confusion_table$FN), sum(confusion_table$FP),  sum(confusion_table$RN)), nrow=2)
#  print(caret::confusionMatrix(as.table(matrix)))
#  print(paste("Youden-Score:", mean(confusion_table$Sensitivitaet, na.rm = T) + mean(confusion_table$Spezifitaet, na.rm = T) -1))
#}

with(confusion_table)

hist(confusion_table$Sensitivitaet)
hist(confusion_table$Spezifitaet)
hist(confusion_table$PPV)
hist(confusion_table$NPV)

#table(confusion_table[,RP==FN])
#uniqueN(zeitintervalle$ID)

#### Sensitivität
print("Sensitivität")
nrow(zeitintervalle[Zeitintervall==0])/uniqueN(zeitintervalle$ID)

#### PPV
print("PPV")
nrow(zeitintervalle[Zeitintervall==0])/(nrow(zeitintervalle[Zeitintervall==0])+nrow(zeitintervalle[Zeitintervall>0][HA1 == 1 | HA2 == 1]))


#### Spezifität
print("Spezifiät")
sum(confusion_table$RN)/(sum(confusion_table$RN)+sum(confusion_table$FN))

#### NPV
print("NPV")
sum(confusion_table$RN)/(sum(confusion_table$RN) + (uniqueN(zeitintervalle$ID) - nrow(zeitintervalle[Zeitintervall==0])))

fwrite(daten_moocall, file = "2018-09-10_daten_moocall.csv")
fwrite(confusion_table, file = "2018-09-10_confusion_table.csv")
fwrite(zeitintervalle, file = "2018-09-10_zeitintervalle_3h.csv")
