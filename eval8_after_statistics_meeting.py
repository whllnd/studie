import csv
import datetime
import numpy as np
import pprint
import matplotlib.pyplot as plt
from copy import deepcopy

np.set_printoptions(precision=2)

heifers = [
    2566,
    2576,
    2650,
    2595,
    2728,
    2625,
    2630,
    2633,
    2636,
    2639,
    2641,
    2658,
    2747,
    2581,
    2712,
    2737,
    2545,
    2558,
    2571,
    2578,
    2596,
    2601,
    2621]

# ==============================================================================

class event:
    def __init__(self): # , start_time, end_time, score):
        self.start = None # Column B or "SonT"
        self.end = None   # Column F or "Event"
        self.score = None
        self.real_calving_time = None
        self.moocall_score = None
        self.sensor_off_tail = None

    def __lt__(self, other):
        return self.start < other.start

class measurement:

    def __init__(self, cow_id, study):
        self.id = cow_id
        self.study = study
        self.events = []
        self.ha1_warnings = []
        self.ha2_warnings = []
        self.carry_time = 0
        self.did_calve = False
        self.is_heifer = self.id in heifers
        self.real_calving_time = None
        self.calving_condition = ""
        if self.is_heifer:
            self.cow_idx = 1
        else:
            self.cow_idx = 0

    def __str__(self):
        s = "\nMeasurement: " + str(self.id)
        for event in self.events:
            s += "\nEvent:"
            s += "\n" + str(event.start)
            s += "\n" + str(event.end)
            if event.real_calving_time is not None:
                s += "\n" + str(event.real_calving_time)
            s += "\n" + str(event.score)
        s += "\nWarnings (ha1):"
        return s

# ==============================================================================

def parse_time(time):
    try:
        dt = datetime.datetime.strptime(time, "%d.%m.%Y %H:%M")
        return dt
    except ValueError:
        print("Failure:", time)
        raise

# ==============================================================================

# Evaluation model: --|start --- threshold --- event|--
# measurement_list is expected to contain only cows that calved and are from the same study
HA1, HA2 = 0, 1
TP, FN, FP, TN = 0, 1, 2, 3
def evaluate(measurement_list, threshold):

    results_cows    = [[0]*4, [0]*4] # [ha1, ha2]
    results_heifers = [[0]*4, [0]*4] # [ha1, ha2]
    totals_cows     = [[0]*4, [0]*4] # Only relevant for tp and fp, though
    totals_heifers  = [[0]*4, [0]*4] # Only relevant for tp and fp, though

    for m in measurement_list:

        # Determine time of calving
        calving_time = None
        for e in m.events:
            if e.score == 1:
                calving_time = deepcopy(e.end)
                break
        assert calving_time is not None, "Calving time is none?"
        #calving_time = [deepcopy(e.end) for e in m.events if e.score == 1][0]
        thresh = calving_time - datetime.timedelta(hours=threshold)
        start = calving_time - datetime.timedelta(hours=48)

        for i,warnings in enumerate([m.ha1_warnings, m.ha2_warnings]):

            # True positives, false negatives, false positives, true negatives
            tp = sum([1 for w in warnings if w >= thresh and w <= calving_time])
            fn = 0 if tp > 0 else 1
            fp = sum([1 for w in warnings if w >= start and w <= thresh])
            tn = 0 if fp > 0 else 1

            # Update cow results
            if not m.is_heifer:
                results_cows[i][TP] += min(1, tp)
                results_cows[i][FN] += fn
                results_cows[i][FP] += min(1, fp)
                results_cows[i][TN] += tn

                # Keep track of total occurences
                totals_cows[i][TP] += tp
                totals_cows[i][FP] += fp
            else:
                results_heifers[i][TP] += min(1, tp)
                results_heifers[i][FN] += fn
                results_heifers[i][FP] += min(1, fp)
                results_heifers[i][TN] += tn

                # Keep track of total occurences
                totals_heifers[i][TP] += tp
                totals_heifers[i][FP] += fp

    return results_cows, totals_cows, results_heifers, totals_heifers

# ==============================================================================

# Evaluation model: --|start --- threshold --- event|--
# measurement_list is expected to contain only cows that calved and are from the same study
def false_alarms(calvings, threshold):

    FP = [0, 0, 0, 0] # ha1_cows, ha1_heifers, ha2_cows, ha2_heifers
    FN = [0, 0, 0, 0]

    for m in calvings:

        # Determine time of calving
        calving_time = None
        for e in m.events:
            if e.score == 1:
                calving_time = deepcopy(e.end)
                break
        assert calving_time is not None, "Calving time is none?"
        thresh = calving_time - datetime.timedelta(hours=threshold)
        start = calving_time - datetime.timedelta(hours=48)

        for i,warnings in enumerate([m.ha1_warnings, m.ha2_warnings]):

            # True positives, false negatives, false positives, true negatives
            tp = sum([1 for w in warnings if w >= thresh and w <= calving_time])
            fn = 0 if tp > 0 else 1
            fp = sum([1 for w in warnings if w >= start and w <= thresh])
            tn = 0 if fp > 0 else 1

            if m.is_heifer:
                FP[i * 2 + 1] += fp
                FN[i * 2 + 1] += fn
            else:
                FP[i * 2] += fp
                FN[i * 2] += fn

    return FP, FN

# ==============================================================================

def csv_export(results_dict, csv_fname):

    # 3 -> [s1, s2, s1+2], 2 -> [HA1, HA2], 6 -> [tp, fn, fp, tn, total_tp, total_fp], len(d.keys()) -> different thresholds
    mat = np.zeros((3,2,6,len(results_dict[1].keys())))
    for study in results_dict:
        d = results_dict[study]
        for i,threshold in enumerate(d):
            results, totals = d[threshold]
            for ha in [HA1, HA2]:
                mat[study-1,ha,TP,i] = results[ha][TP]
                mat[study-1,ha,FN,i] = results[ha][FN]
                mat[study-1,ha,FP,i] = results[ha][FP]
                mat[study-1,ha,TN,i] = results[ha][TN]
                mat[study-1,ha,4,i] = totals[ha][TP]
                mat[study-1,ha,5,i] = totals[ha][FP]

    mat[2,0] = mat[0,0] + mat[1,0] # Total HA1
    mat[2,1] = mat[0,1] + mat[1,1] # Total HA2

    key_map = { 0:"TP", 1:"FN", 2:"FP", 3:"TN", 4:"Total TP", 5:"Total FP" }
    with open(csv_fname, "w") as fh:
        writer = csv.writer(fh, delimiter=";")

        # Let's spare loops and just copy and paste it in hard-coding fashing
        writer.writerow(["Total:"])
        writer.writerow(["HA1"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[2,0,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])
        writer.writerow([""])
        writer.writerow(["HA2"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[2,1,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])

        writer.writerow([""])
        writer.writerow([""])
        writer.writerow(["Studie 1:"])
        writer.writerow(["HA1"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[0,0,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])
        writer.writerow([""])
        writer.writerow(["HA2"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[0,1,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])

        writer.writerow([""])
        writer.writerow([""])
        writer.writerow(["Studie 2:"])
        writer.writerow(["HA1"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[1,0,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])
        writer.writerow(["HA2"])
        writer.writerow(["Threshold [h]"] + [str(t) for t in results_dict[1].keys()])
        for i in range(6):
            writer.writerow([key_map[i]] + [str(mat[1,1,i,thresh]).replace(".",",") for thresh in range(mat.shape[3])])

# ==============================================================================

def process(batch, measurements, study):

    # Ensure that all rows in batch have same cow id
    if not all([batch[0][0] == batch[i][0] for i in range(1, len(batch))]):
        raise Exception("Not identical IDs")

    # Process batch
    id = int(batch[0][A])
    e = event()
    ha1s, ha2s = [], []
    for j,line in enumerate(batch):

        # Handle warnings
        if len(line[C]) > 0:
            ha1s.append(parse_time(line[C]))
        if len(line[D]) > 0:
            ha2s.append(parse_time(line[D]))

        # Decide if we need to keep collecting lines or process batchrent line
        score = int(line[G])
        shall_be_event = score != 6 and score != 7

        if score == 6: # We ignore end times ...

            #if score == 6 and len(line[F]) > 0:
            #    print("Although score 6 we got an end time:", i, id)
            #    raise

            if len(line[B]) > 0: # Supposed to be start time
                start = parse_time(line[B])
                e.start = start
            continue

        #if score != 6 and score != 7: # Excluding the sevens here
        if shall_be_event:

            if len(line[F]) == 0:
                print("Although score not 6 we have no end time:", i, id)
                raise

            # It is possible to have one line events, therefore check for both start and end time
            if len(line[B]) > 0:

                if e.start is not None:
                    print("Should be None, that start time:", i, id, score)
                    raise
                e.start = parse_time(line[B])

            if len(line[F]) > 0:

                if e.end is not None:
                    print("Should be None, that end time:", i, id)
                    raise

                e.end = parse_time(line[F])

            if len(line[E]) > 0:

                if e.sensor_off_tail is not None:
                    print("Should be None, that sensor_off_tail:", i, id)
                    raise

                e.sensor_off_tail = parse_time(line[E])
            else:
                e.sensor_off_tail = ""

        if e.end is None or e.start is None:
            print("Both event times are None for id", id)
            raise

        e.score = score
        if score == 1 and len(line[I]) > 0:
            e.real_calving_time = parse_time(line[I])
        if len(line[H]) > 0:
            try:
                e.moocall_score = int(line[H])
            except ValueError:
                pass
        if id in measurements:
            measurements[id].events.append(deepcopy(e))
            measurements[id].study = study
            measurements[id].ha1_warnings += [deepcopy(h1) for h1 in ha1s] # Be on the sure side and copy
            measurements[id].ha2_warnings += [deepcopy(h2) for h2 in ha2s]
            if len(line[I]) > 0:
                measurements[id].real_calving_time = parse_time(line[I])
        else:
            m = measurement(id, study)
            m.events.append(deepcopy(e))
            m.ha1_warnings += [deepcopy(h1) for h1 in ha1s] # Be on the sure side and copy
            m.ha2_warnings += [deepcopy(h2) for h2 in ha2s]
            if len(line[I]) > 0:
                m.real_calving_time = parse_time(line[I])
            measurements[id] = m

        e = event()
        ha1s, ha2s = [], []


# ==============================================================================

# Open data file
#with open("./Moocall_Daten_korrigiert-08-Feb-2018.csv") as fh:
with open("./ROHDATEN-sortiert-26-Jul-2018.csv") as fh:
    study = list(csv.reader(fh, delimiter=","))

# Index to excel column
A,B,C,D,E,F,G,H,I = 0,1,2,3,4,5,6,7,8

# Read data
measurements = {}
cur = [] # Collection of lines for current cow id
unique_id = [set(), set()]
s1, e1 = datetime.datetime(2030, 1, 1), datetime.datetime(1900, 1, 1)
s2, e2 = datetime.datetime(2030, 1, 1), datetime.datetime(1900, 1, 1)
sevens1 = 0
sevens2 = 0
shall_break = False

for i,r in enumerate(study):
    if i == 0:
        continue # Discard header line
    #print(i)

    if r[G] == "20":
        continue

    # We're done (total file)
    if i >= 963:
        shall_break = True

    # Unique cow id's and global period times
    if not shall_break:
        on_tail = parse_time(r[B]) if len(r[B]) > 0 else None
        off_tail = parse_time(r[E]) if len(r[E]) > 0 else None
        seven = 1 if r[G] == "7" else 0
        if i < 666:
            unique_id[0].add(int(r[A]))
            if on_tail is not None and on_tail < s1:
                s1 = deepcopy(on_tail)
            if off_tail is not None and off_tail > e1:
                e1 = deepcopy(off_tail)
            sevens1 += seven
        else:
            unique_id[1].add(int(r[A]))
            if on_tail is not None and on_tail < s2:
                s2 = deepcopy(on_tail)
            if off_tail is not None and off_tail > e2:
                e2 = deepcopy(off_tail)
            sevens2 += seven

        if r[G] == "":
            print(i)
            raise "Uncool!"

        # Ignore lines where score is not 6 and we have no start nor end time
        if r[G] != "6" and len(r[B]) == 0 and len(r[F]) == 0:
            #print("Line with no 6, no start and no end time...", i)
            continue

    # Collect batch based on same cow id
    if len(cur) == 0 and i < len(study):
        cur.append(r)
        continue

    # As long as we see same cow id, we keep collecting
    if r[0] == cur[-1][0] and i < len(study):
        cur.append(r)
        continue

    process(cur, measurements, 1 if i < 666 else 2)

    if shall_break:
        #print("Last pack:")
        #pprint.pprint(cur)
        break

    # Reset current line collection and move on to next batch
    cur = [r]

#for id in measurements:
#    print("\nMeasurement:", id)
#    m = measurements[id]
#    for event in m.events:
#        print("Event:")
#        print("   ", event.start)
#        print("   ", event.end)
#        print("   ", event.score)
#    print("Warnings (ha1):")
#    for w in m.ha1_warnings:
#        print("   ", w)
#    print("Warnings (ha2):")
#    for w in m.ha2_warnings:
#        print("   ", w)
#    input("Next...")

# Sanity: each cow should only have one calving!
calvings = [0, 0] # cows, heifers
for id in measurements:
    m = measurements[id]
    c = 0
    carry_time = 0
    for e in m.events:
        if e.score == 1:
            c += 1
        if e.end is None or e.start is None:
            continue
        dt = (e.end - e.start).total_seconds() / 3600
        if dt < 0:
            print("Outrageous: " + str(id))
            print(e.end)
            print(e.start)
        carry_time += dt

    if c > 1:
        print(m)
        raise Exception("OUTRAGEOUS!")

    if c > 0:
        m.did_calve = True

    calvings[m.cow_idx] += c
    m.carry_time = carry_time

# Sanity: no event should overlap in time with another event from the same cow
for id in measurements:
    events = measurements[id].events
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            if events[i].end > events[j].start and events[i].start < events[j].end:
                print("No cool:", id)
                #print([(e.start, e.end) for e in events])
            if events[i].start > events[j].start and events[i].end < events[j].end:
                print("2 No cool:", id)

# Sanity: no warning should be issued within one hour after a warning
for id in measurements:
    warnings1 = measurements[id].ha1_warnings
    for i in range(len(warnings1)-1):
        dt = abs((warnings1[i+1] - warnings1[i]).total_seconds() / 3600)
        if dt < 1.:
            print("Ne, oder!?")
            print(warnings1[i])
            print(warnings1[i+1])
    warnings2 = measurements[id].ha2_warnings
    for i in range(len(warnings2)-1):
        dt = abs((warnings2[i+1] - warnings2[i]).total_seconds() / 3600)
        if dt < 1.:
            print("Ne, oder!? 2222")
            print(warnings2[i])
            print(warnings2[i+1])

# Remove animals with event 8
remove = []
for id in measurements:
    if 0 < len([e for e in measurements[id].events if 8 == e.score]):
        remove.append(id)
for id in remove:
    del measurements[id]

maxHA1, maxHA2 = 0, 0
for id in measurements:
    maxHA1 = max(maxHA1, len(measurements[id].ha1_warnings))
    maxHA2 = max(maxHA2, len(measurements[id].ha2_warnings))

# Sort events and ensure sanity
for id in measurements:
    m = measurements[id]
    m.events = sorted(m.events, key=lambda e: e.start)
    for i in range(1, len(m.events)):
        if m.events[i].start < m.events[i-1].end:
            print(id, e.start, e.end)
    if m.did_calve:
        if m.events[-1].score != 1:
            print(id, m.events[-1])

# Evaluation ###################################################################

# Verteilungskurve Events
with open("01_Verteilungskurve_der_Events.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Anzahl Event 1", "Anzahl Event 2", "Anzahl Event 3", "Anzahl Event 4", "Anzahl Event 5", "Anzahl Event 6"])
    for id in measurements:
        m = measurements[id]
        f.writerow([id,
                    len([e for e in m.events if e.score == 1]),
                    len([e for e in m.events if e.score == 2]),
                    len([e for e in m.events if e.score == 3]),
                    len([e for e in m.events if e.score == 4]),
                    len([e for e in m.events if e.score == 5]),
                    len([e for e in m.events if e.score == 6])])

# Zeiträume zwischen Alarm und Umstallzeit
with open("02_Zeitraeume_Alarm_Onset_Stage_II.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Kuh / Faerse", "Gekalbt (ja / nein)", "Alarm Qualitaet", "Alarm Zeitpunkt", "Umstallzeit", "Zeitraum Alarm-Umstallzeit [h]"])
    for id in measurements:
        m = measurements[id]
        status = "Faerse" if m.is_heifer else "Kuh"
        calved = "ja" if m.did_calve else "nein"
        tumstall = ""
        for e in m.events:
            if e.score == 1:
                tumstall = e.end
        for w in m.ha1_warnings:
            dt = ""
            if m.did_calve:
                dt = str((tumstall - w).total_seconds() / 3600.).replace(".", ",")
            f.writerow([id, status, calved, "HA1h", w, tumstall, dt])
        for w in m.ha2_warnings:
            dt = ""
            if m.did_calve:
                dt = str((tumstall - w).total_seconds() / 3600.).replace(".", ",")
            f.writerow([id, status, calved, "HA2h", w, tumstall, dt])

# Sensitivität der Alarme
numFirstHA1, numFirstHA2 = 0, 0
numHA1, numHA2 = 0, 0
numCalvings = 0
tpHA1 = [0,0,0] # 1h, 2h, 3h threshold
tpHA2 = [0,0,0] # 1h, 2h, 3h threshold
animalsWithNoHA1, animalsWithNoHA2 = 0, 0
with open("03_Sensitivitaet_Alarme.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    header = ["Kuh ID", "Kuh / Faerse", "Gekalbt (ja / nein)", "Erster SonT", "Anzahl HA1-Alarme"]
    for i in range(maxHA1):
        header.append("HA1_" + str(i + 1))
    header.append("Anzahl HA2-Alarme")
    for i in range(maxHA2):
        header.append("HA2_" + str(i + 1))
    header.append("Umstallzeit")
    for i in range(maxHA1):
        header.append("dt HA1_" + str(i + 1) + " [h]")
    for i in range(maxHA2):
        header.append("dt HA2_" + str(i + 1) + " [h]")
    f.writerow(header)

    for id in measurements:
        m = measurements[id]
        status = "Faerse" if m.is_heifer else "Kuh"
        calved = "ja" if m.did_calve else "nein"
        tumstall = ""
        sont = m.events[0].start
        for e in m.events:
            sont = min(sont, e.start)
            if e.score == 1:
                tumstall = e.end
                numCalvings += 1
        ha1 = [""] * maxHA1
        ha2 = [""] * maxHA2
        dt_ha1 = [""] * maxHA1
        dt_ha2 = [""] * maxHA2
        for i,w in enumerate(m.ha1_warnings):
            ha1[i] = w
            if m.did_calve:
                dt_ha1[i] = str((tumstall - w).total_seconds() / 3600.).replace(".", ",")
        for i,w in enumerate(m.ha2_warnings):
            ha2[i] = w
            if m.did_calve:
                dt_ha2[i] = str((tumstall - w).total_seconds() / 3600.).replace(".", ",")
        f.writerow([id, status, calved, sont, len(m.ha1_warnings)] + ha1 + [len(m.ha2_warnings)] + ha2 + [tumstall] + dt_ha1 + dt_ha2)

        if 0 < len(m.ha1_warnings):
            numFirstHA1 += 1
        else:
            animalsWithNoHA1 += 1
        if 0 < len(m.ha2_warnings):
            numFirstHA2 += 1
        else:
            animalsWithNoHA2 += 1
        numHA1 += len(m.ha1_warnings)
        numHA2 += len(m.ha2_warnings)

        if m.did_calve:
            if 0 < len(m.ha1_warnings):
                firstHA1 = m.ha1_warnings[0]
                for w in m.ha1_warnings:
                    firstHA1 = min(firstHA1, w)
                dt = (tumstall - firstHA1).total_seconds() / 3600.
                if dt < 0.:
                    print("Erster HA1 Alarm nach Umstallung:", dt, "id:", id)
                else:
                    if dt < 1.:
                        tpHA1[0] += 1
                    if dt < 2.:
                        tpHA1[1] += 1
                    if dt < 3.:
                        tpHA1[2] += 1

            if 0 < len(m.ha2_warnings):
                firstHA2 = m.ha2_warnings[0]
                for w in m.ha2_warnings:
                    firstHA2 = min(firstHA2, w)
                dt = (tumstall - firstHA2).total_seconds() / 3600.
                if dt < 0.:
                    print("Erster HA2 Alarm nach Umstallung:", dt, "id:", id)
                else:
                    if dt < 1.:
                        tpHA2[0] += 1
                    if dt < 2.:
                        tpHA2[1] += 1
                    if dt < 3.:
                        tpHA2[2] += 1

print("03. Sensitivitaet Alarme")
print("Anzahl aller HA1-Alarme (180 Tiere):", numHA1)
print("Anzahl aller HA2-Alarme (180 Tiere):", numHA2)
print("Anzahl aller Geburten (180 Tiere):  ", numCalvings)
print("Verhältnis aller HA1-Alarme zur Anzahl aller Geburten:", numHA1 / numCalvings)
print("Verhältnis aller HA2-Alarme zur Anzahl aller Geburten:", numHA2 / numCalvings)
print("Anzahl aller ersten HA1-Alarme:", numFirstHA1)
print("Anzahl aller ersten HA2-Alarme:", numFirstHA2)
print("Tiere ohne HA1 Alarm:", animalsWithNoHA1)
print("Tiere ohne HA2 Alarm:", animalsWithNoHA2)
print("TP HA1 (1h):", tpHA1[0], "Verhältnis zu allen ersten HA1-Alarme:", tpHA1[0] / numFirstHA1)
print("TP HA1 (2h):", tpHA1[1], "Verhältnis zu allen ersten HA1-Alarme:", tpHA1[1] / numFirstHA1)
print("TP HA1 (3h):", tpHA1[2], "Verhältnis zu allen ersten HA1-Alarme:", tpHA1[2] / numFirstHA1)
print("---")
print("TP HA2 (1h):", tpHA2[0], "Verhältnis zu allen ersten HA2-Alarme:", tpHA2[0] / numFirstHA2)
print("TP HA2 (2h):", tpHA2[1], "Verhältnis zu allen ersten HA2-Alarme:", tpHA2[1] / numFirstHA2)
print("TP HA2 (3h):", tpHA2[2], "Verhältnis zu allen ersten HA2-Alarme:", tpHA2[2] / numFirstHA2)
print("")

# Tabelle 1 (misc)
hours_all, hours_calved = [], []
time_before_calving = [0,0,[],[]] # carry time before umstall, umstall-geburt, ha1 calving, ha2 calving
time_before_calving_cows = [0,0,[],[]]
time_before_calving_heif = [0,0,[],[]]
first_ha1_all, first_ha2_all = [], []
first_ha1_cows, first_ha2_cows = [], []
first_ha1_heif, first_ha2_heif = [], []
last_ha1_all, last_ha2_all = [], []
last_ha1_cows, last_ha2_cows = [], []
last_ha1_heif, last_ha2_heif = [], []
events = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for id in measurements:
    m = measurements[id]
    tumstall = None
    for e in m.events:
        events[e.score] += 1
        dt = (e.end - e.start).total_seconds() / 3600.
        if dt < 0:
            print(id, e.start, e.end)
            raise
        hours_all.append(dt)
        if m.did_calve:
            hours_calved.append(dt)
            if e.score == 1:
                tumstall = e.end

    if m.did_calve:

        min_dt_ha1, max_dt_ha1 = np.inf, -np.inf
        for w in m.ha1_warnings:
            dt = (tumstall-w).total_seconds() / 3600.
            if dt >= 0.:
                time_before_calving[2].append(dt)
                if m.is_heifer:
                    time_before_calving_heif[2].append(dt)
                else:
                    time_before_calving_cows[2].append(dt)
                min_dt_ha1 = min(min_dt_ha1, dt)
                max_dt_ha1 = max(max_dt_ha1, dt)

        min_dt_ha2, max_dt_ha2 = np.inf, -np.inf
        for w in m.ha2_warnings:
            dt = (tumstall-w).total_seconds() / 3600.
            if dt >= 0.:
                time_before_calving[3].append(dt)
                if m.is_heifer:
                    time_before_calving_heif[3].append(dt)
                else:
                    time_before_calving_cows[3].append(dt)
                min_dt_ha2 = min(min_dt_ha2, dt)
                max_dt_ha2 = max(max_dt_ha2, dt)

        if min_dt_ha1 < np.inf and min_dt_ha2 < np.inf:
            first_ha1_all.append(max_dt_ha1)
            last_ha1_all.append(min_dt_ha1)
            first_ha2_all.append(max_dt_ha2)
            last_ha2_all.append(min_dt_ha2)
        if m.is_heifer:
            if min_dt_ha1 < np.inf and min_dt_ha2 < np.inf:
                first_ha1_heif.append(max_dt_ha1)
                last_ha1_heif.append(min_dt_ha1)
                first_ha2_heif.append(max_dt_ha2)
                last_ha2_heif.append(min_dt_ha2)
        else:
            if min_dt_ha1 < np.inf and min_dt_ha2 < np.inf:
                first_ha1_cows.append(max_dt_ha1)
                last_ha1_cows.append(min_dt_ha1)
                first_ha2_cows.append(max_dt_ha2)
                last_ha2_cows.append(min_dt_ha2)

    if m.did_calve:
        time_before_calving[0] += sum([(e.end - e.start).total_seconds() / 3600. for e in m.events])
        time_before_calving[1] += (m.events[-1].real_calving_time - m.events[-1].end).total_seconds() / 3600.
        if m.is_heifer:
            time_before_calving_heif[0] += sum([(e.end - e.start).total_seconds() / 3600. for e in m.events])
            time_before_calving_heif[1] += (m.events[-1].real_calving_time - m.events[-1].end).total_seconds() / 3600.
        else:
            time_before_calving_cows[0] += sum([(e.end - e.start).total_seconds() / 3600. for e in m.events])
            time_before_calving_cows[1] += (m.events[-1].real_calving_time - m.events[-1].end).total_seconds() / 3600.

print("Number of animals / cows / heifers:", 180, 180 - len(heifers), len(heifers))
print("Number of calvings:", len([id for id in measurements if measurements[id].did_calve]))
print("Total of hours monitored (180 animals): ", sum(hours_all))
print("Total of hours monitored (118 calvings):", sum(hours_calved))
print("For all animals:")
print("Average time [h] sensor on tail before stage II: ", time_before_calving[0] / 118)
print("Average time [h] between stage II and real birth:", time_before_calving[1] / 118)
print("Average time [h] between HA1 and stage II (std): ", np.array(time_before_calving[2]).mean(), "(", np.array(time_before_calving[2]).std(), ")")
print("Average time [h] first HA1 and stage II (std):   ", np.array(first_ha1_all).mean(), "(", np.array(first_ha1_all).std(), ")")
print("Average time [h] last HA1 and stage II (std):    ", np.array(last_ha1_all).mean(), "(", np.array(last_ha1_all).std(), ")")
print("Average time [h] between HA2 and stage II (std): ", np.array(time_before_calving[3]).mean(), "(", np.array(time_before_calving[3]).std(), ")")
print("Average time [h] first HA2 and stage II (std):   ", np.array(first_ha2_all).mean(), "(", np.array(first_ha2_all).std(), ")")
print("Average time [h] last HA2 and stage II (std):    ", np.array(last_ha2_all).mean(), "(", np.array(last_ha2_all).std(), ")")
print("For cows:")
print("Average time [h] sensor on tail before stage II: ", time_before_calving_cows[0] / 95)
print("Average time [h] between stage II and real birth:", time_before_calving_cows[1] / 95)
print("Average time [h] between HA1 and stage II (std): ", np.array(time_before_calving_cows[2]).mean(), "(", np.array(time_before_calving_cows[2]).std(), ")")
print("Average time [h] first HA1 and stage II (std):   ", np.array(first_ha1_cows).mean(), "(", np.array(first_ha1_cows).std(), ")")
print("Average time [h] last HA1 and stage II (std):    ", np.array(last_ha1_cows).mean(), "(", np.array(last_ha1_cows).std(), ")")
print("Average time [h] between HA2 and stage II (std): ", np.array(time_before_calving_cows[3]).mean(), "(", np.array(time_before_calving_cows[3]).std(), ")")
print("Average time [h] first HA2 and stage II (std):   ", np.array(first_ha2_cows).mean(), "(", np.array(first_ha2_cows).std(), ")")
print("Average time [h] last HA2 and stage II (std):    ", np.array(last_ha2_cows).mean(), "(", np.array(last_ha2_cows).std(), ")")
print("For heifers:")
print("Average time [h] sensor on tail before stage II: ", time_before_calving_heif[0] / 23)
print("Average time [h] between stage II and real birth:", time_before_calving_heif[1] / 23)
print("Average time [h] between HA1 and stage II (std): ", np.array(time_before_calving_heif[2]).mean(), "(", np.array(time_before_calving_heif[2]).std(), ")")
print("Average time [h] first HA1 and stage II (std):   ", np.array(first_ha1_heif).mean(), "(", np.array(first_ha1_heif).std(), ")")
print("Average time [h] last HA1 and stage II (std):    ", np.array(last_ha1_heif).mean(), "(", np.array(last_ha1_heif).std(), ")")
print("Average time [h] between HA2 and stage II (std): ", np.array(time_before_calving_heif[3]).mean(), "(", np.array(time_before_calving_heif[3]).std(), ")")
print("Average time [h] first HA2 and stage II (std):   ", np.array(first_ha2_heif).mean(), "(", np.array(first_ha2_heif).std(), ")")
print("Average time [h] last HA2 and stage II (std):    ", np.array(last_ha2_heif).mean(), "(", np.array(last_ha2_heif).std(), ")")
print("Risk periods:")
print("Total number of 1 hour RPs:", sum(hours_all)) # Should be the same
print("#Event 2 (share):          ", events[2], "(", events[2] / sum(hours_all), ")")
print("#Event 3 (share):          ", events[3], "(", events[3] / sum(hours_all), ")")
print("#Event 4 (share):          ", events[4], "(", events[4] / sum(hours_all), ")")
print("#Event 5 (share):          ", events[5], "(", events[5] / sum(hours_all), ")")
print("Events:", events)

verlauf = {}
with open("./Voss-Moocall-Geburtsverlauf_Korrelation_Dystokie_Kuh_Faerse_24-Jul-2018.csv") as fh:
    data = list(csv.reader(fh, delimiter=";"))[1:]
    for d in data:
        id = int(d[0])
        cond = int(d[3])
        measurements[id].calving_condition = cond

# Export data in a clean way
with open("Rohdaten_2018-09-05.csv", "w") as fh:

    f = csv.writer(fh, delimiter=";")
    f.writerow(["ID", "Kuh/Faerse",
                    "Gekalbt (ja/nein)", "Umstallzeit", "Echte Geburtszeit", "Geburtsverlauf",
                    "Event (ja/nein)", "Event Score", "SonT", "Eventzeit", "SoffT", "Moocall Score",
                    "Alarm (ja/nein)", "Alarmtyp (HA1/HA2)", "Alarmzeit"])

    for id in measurements:

        m = measurements[id]
        status = "Faerse" if m.is_heifer else "Kuh"
        calved = "ja" if m.did_calve else "nein"
        tpart = "" if not m.did_calve else m.events[-1].end
        if m.did_calve:
            assert m.events[-1].score == 1
        #tbirth = "" if m.real_calving_time is None else m.real_calving_time
        if m.real_calving_time is None:
            tbirth = m.events[-1].end + datetime.timedelta(hours=5)
        else:
            tbirth = m.real_calving_time

        for e in m.events:
            f.writerow([id, status, calved, tpart, tbirth, m.calving_condition, "ja", e.score, e.start, e.end, e.sensor_off_tail, e.moocall_score, "nein", "", ""])
        for w in m.ha1_warnings:
            f.writerow([id, status, calved, tpart, tbirth, m.calving_condition, "nein", "", "", "", "", "", "ja", "HA1h", w])
        for w in m.ha2_warnings:
            f.writerow([id, status, calved, tpart, tbirth, m.calving_condition, "nein", "", "", "", "", "", "ja", "HA2h", w])

print(len(measurements))
for id in measurements:
    m = measurements[id]
    if m.real_calving_time is None:
        print(id)
