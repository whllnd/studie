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
        else:
            m = measurement(id, study)
            m.events.append(deepcopy(e))
            m.ha1_warnings += [deepcopy(h1) for h1 in ha1s] # Be on the sure side and copy
            m.ha2_warnings += [deepcopy(h2) for h2 in ha2s]
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

# Store stuff in csv file
#with open("kuehe_und_kalbungen.csv", "w") as fh:
#    cows = csv.writer(fh, delmiter=",")
#    cows.writerow(["Cow ID", ""
#    for id in measurements:


# Evaluation

# 1.) How many cows were involved
in_study = [0,0] # cows, heifers
for id in measurements:
    m = measurements[id]
    in_study[m.cow_idx] += 1
print("01.) How many cows were involved in the stuy (total):   ", len(unique_id[0]) + len(unique_id[1]))
print("                                             (cows):    ", in_study[0] + 1)
print("                                             (heifers): ", in_study[1])

# 2.) How many cows that were involved had calvings
print("02.) How many cows actually calved (total):   ", sum(calvings))
print("                                   (cows):    ", calvings[0])
print("                                   (heifers): ", calvings[1])

# 3.) Average carrying time for calving cows
act = np.array([measurements[id].carry_time for id in measurements if measurements[id].did_calve])
act_cows = np.array([measurements[id].carry_time for id in measurements if measurements[id].did_calve and measurements[id].cow_idx == 0])
act_heifers = np.array([measurements[id].carry_time for id in measurements if measurements[id].did_calve and measurements[id].cow_idx == 1])
act[act > 96] = 96 # BAD! DO NOT TELL ANYONE!
act_cows[act_cows > 96] = 96
act_heifers[act_heifers > 96] = 96
print("03.) Average device carrying time in hours (total):   ", act.mean(), "(mean)", act.std(), "(std)")
print("                                           (cows):    ", act_cows.mean(), "(mean)", act_cows.std(), "(std)")
print("                                           (heifers): ", act_heifers.mean(), "(mean)", act_heifers.std(), "(std)")

#3.b) Histogram of carry times
carry_times1 = [measurements[id].carry_time for id in measurements if measurements[id].study == 1 and measurements[id].did_calve]
carry_times2 = [measurements[id].carry_time for id in measurements if measurements[id].study == 2 and measurements[id].did_calve]
hist1 = np.histogram(carry_times1, bins=120, range=(0,120))
hist2 = np.histogram(carry_times2, bins=120, range=(0,120))

#for h in range(1, 120):
#    print("between", h-1, "and", h, "hours:")
#    for id in measurements:
#        if not measurements[id].did_calve:
#            continue
#        if measurements[id].carry_time >= h -1 and measurements[id].carry_time < h:
#            print(id, measurements[id].did_calve)

#plt.hist(carry_times1 + carry_times2, bins=60, range=(0,120))
#plt.hist(carry_times2, bins=120, range=(0,120))
#plt.show()

#print(len([measurements[id].carry_time for id in measurements if measurements[id].carry_time >= 24 and measurements[id].did_calve]))

#3.c) How many warnings have been sent how many hours before calving
num_warnings_dict1 = {}
num_warnings_dict2 = {}
for id in measurements:
    m = measurements[id]
    if not m.did_calve:
        continue

    # Get calving time and compare with each warning
    calving_time = [e.end for e in m.events if e.score == 1][0]
    for w in m.ha1_warnings:
        h = int((calving_time - w).total_seconds() / 3600)
        #if h > 120:
        #    print(h, calving_time, w, id)
        if h in num_warnings_dict1:
            num_warnings_dict1[h] += 1
        else:
            num_warnings_dict1[h] = 1
    for w in m.ha2_warnings:
        h = int((calving_time - w).total_seconds() / 3600)
        #if h > 120:
        #    print(h, calving_time, w, id)
        if h in num_warnings_dict2:
            num_warnings_dict2[h] += 1
        else:
            num_warnings_dict2[h] = 1

keys1 = sorted(num_warnings_dict1)
keys2 = sorted(num_warnings_dict2)

with open("anzahl_warnungen_pro_stunde.csv", "w") as fh:
    writer = csv.writer(fh, delimiter=";")
    writer.writerow(["HA1"])
    writer.writerow(["Hours"] + keys1)
    writer.writerow(["Number of warnings"] + [num_warnings_dict1[h] for h in keys1])
    writer.writerow([""])
    writer.writerow(["HA2"])
    writer.writerow(["Hours"] + keys2)
    writer.writerow(["Number of warnings"] + [num_warnings_dict2[h] for h in keys2])

# 4.) Average time between F and I
f_to_i = [0,0]
f_to_i_cows = []
f_to_i_heif = []
for id in measurements:
    for e in measurements[id].events:
        if e.real_calving_time is not None:
            dt = (e.real_calving_time - e.end).total_seconds() / 3600
            #if dt > 5:
            #    print("Big dt:", id, dt)
            #    print("    end time: ", e.end)
            #    print("    real time:", e.real_calving_time)
            if dt < 0:
                #print("    Negative dt for id", id, "dt:", dt)
                #print("        end time: ", e.end)
                #print("        real time:", e.real_calving_time)
                continue # TODO
            f_to_i[measurements[id].study-1] += dt
            if measurements[id].id in heifers:
                f_to_i_heif.append(dt)
            else:
                f_to_i_cows.append(dt)
print("04.) Average time between F and I (total):   ", sum(f_to_i) / sum(calvings))
print("                                  (cows):    ", np.array(f_to_i_cows).mean())
print("                                  (heifers): ", np.array(f_to_i_heif).mean())

# 4.b) Anzahl HA1 und HA2 bei Faersen und Kuehen
ha1 = [0,0] # [cows, heifers]
ha2 = [0,0]
for id in measurements:
    m = measurements[id]
    if id in heifers:
        ha1[1] += len(m.ha1_warnings)
        ha2[1] += len(m.ha2_warnings)
    else:
        ha1[0] += len(m.ha1_warnings)
        ha2[0] += len(m.ha2_warnings)
print("04b) Number of warnings per cows and heifers:")
print("HA1:", ha1[0], "(cows)", ha1[1], "(heifers)")
print("HA2:", ha2[0], "(cows)", ha2[1], "(heifers)")

# 5./6./7.) True positives, false positives, false negatives (TODO: fn = calvings - tp??)
eval_results_cows = {1 : {}, 2 : {}} # {studie1, studie2}
eval_results_heifers = {1 : {}, 2 : {}} # {studie1, studie2}
for study in range(1,3):
    ml = [measurements[id] for id in measurements if measurements[id].did_calve and measurements[id].study == study]
    for threshold in [2,3,6]: # --|start --- threshold --- event|--
        results_cows, totals_cows, results_heifers, totals_heifers = evaluate(ml, threshold)
        eval_results_cows[study][threshold] = [results_cows, totals_cows]
        eval_results_heifers[study][threshold] = [results_heifers, totals_heifers]
csv_export(eval_results_cows, "evaluation_results_cows.csv")
csv_export(eval_results_heifers, "evaluation_results_heifers.csv")

#b72_ha1 = [0,0]
#b72_ha2 = [0,0]
#print("\nThese are all warnings at least 72h before calving:")
#for study in [1,2]:
#    for id in measurements:
#        m = measurements[id]
#        if m.study != study or not m.did_calve:
#            continue
#
#        calving_time = None
#        for e in m.events:
#            if e.score == 1:
#                calving_time = deepcopy(e.end)
#        assert calving_time is not None, "No kiddin'!"
#        start = calving_time - datetime.timedelta(hours=72)
#
#        b72_ha1[study-1] += sum([1 for w in m.ha1_warnings if w < start])
#        b72_ha2[study-1] += sum([1 for w in m.ha2_warnings if w < start])
#
#        #for w in m.ha1_warnings:
#        #    if w < start:
#        #        print("cow id:", m.id, "calving:", calving_time, "72h before:", start, "ha1 warning:", w)
#        #for w in m.ha2_warnings:
#        #    if w < start:
#        #        print("cow id:", m.id, "calving:", calving_time, "72h before:", start, "ha2 warning:", w)

print("05.) See exported csv files \"true_positives.csv\", \"false_positives.csv\", \"true_negatives.csv\", \"false_negatives.csv\"")
#print("05.b) Warnings at least 72h before calving (total):  ", sum(b72_ha1), "(ha1)", sum(b72_ha2), "(ha2)")
#print("                                           (studie1):", b72_ha1[0], "(ha1)", b72_ha2[0], "(ha2)")
#print("                                           (studie2):", b72_ha1[1], "(ha1)", b72_ha2[1], "(ha2)")

# 11.) Number of different events
noe_cows = { 1:0,2:0,3:0,4:0,5:0,7:sevens1,8:0,9:0 }
noe_heifers = { 1:0,2:0,3:0,4:0,5:0,7:sevens2,8:0,9:0 }
for id in measurements:
    for e in measurements[id].events:
        if e.score not in noe_cows:
            print("This key is not there!?", e.score)
            continue
        if measurements[id].cow_idx == 0:
            noe_cows[e.score] += 1
        elif measurements[id].cow_idx == 1:
            noe_heifers[e.score] += 1
        else:
            raise Exception("MOTHER OF GOD!")
print("11.) Number of different events (total = cows + heifers):")
for score in noe_cows:
    print("    ", score, "->", noe_cows[score] + noe_heifers[score], "=", noe_cows[score], "+", noe_heifers[score])
print("In percentage (however, small sample size for heifers):")
for score in noe_cows:
    print("    ", score, "->", noe_cows[score] / 95, "per cow and", noe_heifers[score] / 23, "per heifer")

# 12.) Average time between warnings and F and I, respectively
avg_time_ha1 = [0,0] # cows, heifers
avg_time_ha2 = [0,0]
count_ha1 = [0,0]
count_ha2 = [0,0]
for id in measurements:
    m = measurements[id]
    if not m.did_calve:
        continue

    calving_time = None
    start = None
    for e in m.events:
        if e.score == 1:
            calving_time = deepcopy(e.end)
            start = deepcopy(e.start)
    assert calving_time is not None, "lol"

    for w in m.ha1_warnings:
        if w >= start and w <= calving_time:
            dt = (calving_time - w).total_seconds() / 3600
            #if dt < 0:
            #    print("Negative time difference for id", id, dt)
            #    continue
            #print("ha1:", dt, id)
            if id == 3588:
                print("calving time:", calving_time)
                print("ha1:         ", w)
                print("dt:          ", dt)
            avg_time_ha1[m.cow_idx] += dt
            count_ha1[m.cow_idx] += 1

    for w in m.ha2_warnings:
        if w >= start and w <= calving_time:
            dt = (calving_time - w).total_seconds() / 3600
            #if dt < 0:
            #    print("Negative time difference for id", id, dt)
            #    continue
            #print("ha2:", dt, id)
            if id == 3588:
                print("calving time:", calving_time)
                print("ha2:         ", w)
                print("dt:          ", dt)
            avg_time_ha2[m.cow_idx] += dt
            count_ha2[m.cow_idx] += 1

print("12.) Average time between warnings and calving in hours (total):  ", sum(avg_time_ha1) / sum(count_ha1), "(ha1)", sum(avg_time_ha2) / sum(count_ha2), "(ha2)")
print("                                                        (cows):   ", avg_time_ha1[0] / count_ha1[0], "(ha1)", avg_time_ha2[0] / count_ha2[0], "(ha2)")
print("                                                        (heifers):", avg_time_ha1[1] / count_ha1[1], "(ha1)", avg_time_ha2[1] / count_ha2[1], "(ha2)")

# 13.) Monitoring period
mp1 = (e1-s1).total_seconds() / 3600
mp2 = (e2-s2).total_seconds() / 3600
print("13.) Monitoring period in hours (total):   ", mp1 + mp2)
print("                                (studie 1):", mp1)
print("                                (studie 2):", mp2)

# A few chi square tests; mainly heifers vs cows
#         Heifers Cows
# 1.) FP
#     FN
h = [measurements[id] for id in measurements if measurements[id].did_calve and id in heifers]
c = [measurements[id] for id in measurements if measurements[id].did_calve and id not in heifers]
print("Chi-Sq-Base Cows:")
print(evaluate(c, 3))
print(evaluate(c, 6))
print(evaluate(c, 12))
print(evaluate(c, 24))
print("Chi-Sq-Base Heifers")
print(evaluate(h, 3))
print(evaluate(h, 6))
print(evaluate(h, 12))
print(evaluate(h, 24))

# Some additional stuff from nearly end of july
print("\n14.) How many false alarms")
calvings = [measurements[id] for id in measurements if measurements[id].did_calve]
for thresh in [2,3,6]:
    fp, fn = false_alarms(calvings, thresh) # [ha1_cows, ha1_heifers, ha2_cows, ha2_heifers]
    fp = np.array(fp)
    fn = np.array(fn)
    fpp = fp / np.array([95, 23, 95, 23])
    fnp = fn / np.array([95, 23, 95, 23])
    print("Thresh:", thresh)
    print("FP HA1:", fpp[0], "per cow vs", fpp[1], "per heifer (absolut:", fp[0], "vs", fp[1],")")
    print("FP HA2:", fpp[2], "per cow vs", fpp[3], "per heifer (absolut:", fp[2], "vs", fp[3],")")
    print("FN HA1:", fnp[0], "per cow vs", fnp[1], "per heifer (absolut:", fn[0], "vs", fn[1],")")
    print("FN HA2:", fnp[2], "per cow vs", fnp[3], "per heifer (absolut:", fn[2], "vs", fn[3],")")


# 15.) How often HA1 after first SonT -> Abwehr?
print("\n15.) After SonT, how often occurs an HA1 message within first 2h?")
dt_cows = []
dt_heifers = []
for m in [measurements[id] for id in measurements if measurements[id].did_calve]:
    if len(m.ha1_warnings) == 0:
        continue
    sorted_events = sorted(m.events)
    sorted_ha1 = sorted(m.ha1_warnings)
    d = (sorted_ha1[0] - sorted_events[0].start).total_seconds()
    if d < 0:
        print("id:", m.id, "1st event:", sorted_events[0].start, "1st ha1:", sorted_ha1[0])
        raise Exception("15.)")
        continue

    for e in sorted_events:
        min_dt = np.inf
        for w in sorted_ha1:
            delta = (w - e.start).total_seconds()
            if delta > 0 and delta <= 2*3600 and delta < min_dt:
                min_dt = delta

        if min_dt == np.inf:
            continue

        if m.is_heifer:
            dt_heifers.append(min_dt)
        else:
            dt_cows.append(min_dt)

dt = dt_cows + dt_heifers
within_cows = len([d for d in dt_cows if d <= 2 * 3600])
within_heifers = len([d for d in dt_heifers if d <= 2 * 3600])
print("Cows + heifers:")
print("   Number of times HA1 occured within first 2h after SonT:", len([d for d in dt if d <= 2 * 3600]))
print("   Average time in hours between HA1 and SonT if dt <= 2h:", np.array([d for d in dt if d <= 2 * 3600]).mean() / 3600)
print("Cows:")
print("   Number of times HA1 occured within first 2h after SonT:", within_cows, "abs,", within_cows / 95, "per cow")
print("   Average time in hours between HA1 and SonT if dt <= 2h:", np.array([d for d in dt_cows if d <= 2 * 3600]).mean() / 3600)
print("Heifers:")
print("   Number of times HA1 occured within first 2h after SonT:", within_heifers, "abs,", within_heifers / 23, "per heifer")
print("   Average time in hours between HA1 and SonT if dt <= 2h:", np.array([d for d in dt_heifers if d <= 2 * 3600]).mean() / 3600)


with open("./Voss-Moocall-Geburtsverlauf_Korrelation_Dystokie_24-Jul-2018.csv") as fh:
    verlauf = list(csv.reader(fh, delimiter=";"))[1:]

verlaufs = {}
for v in verlauf:
    id = int(v[0])
    geburtsverlauf = int(v[3])
    verlaufs[id] = geburtsverlauf

# CSV with all HA1 and HA2 stuff
with open("HA1_HA2_warnings_48h.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Typ (HA1 / HA2)", "Kuh ID", "Status (Kuh / Faerse)", "Zeitstempel", "Umstallzeit", "Geburtszeit", "Geburtsverlauf"])
    for id in measurements:
        if not measurements[id].did_calve:
            continue

        m = measurements[id]
        status = "Kuh" if not m.is_heifer else "Faerse"
        geburtsverlauf = verlaufs[id]
        tumstall, tgeburt = None, None
        for e in m.events:
            if e.score == 1:
                tumstall = e.end
                tgeburt = e.real_calving_time

        for w in m.ha1_warnings:
            if (tumstall - w).total_seconds() / 3600. > 48.:
                continue
            f.writerow(["HA1", str(id), status, str(w), str(tumstall), str(tgeburt), geburtsverlauf])

        for w in m.ha2_warnings:
            if (tumstall - w).total_seconds() / 3600. > 48.:
                continue
            f.writerow(["HA2", str(id), status, str(w), str(tumstall), str(tgeburt), geburtsverlauf])

# Jeweils letzte Alarme
with open("Last_HA1_HA2_warnings_per_animal_48h.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Status (Kuh=1 / Faerse=0)", "Anzahl HA1 Alarme (48h)", "Anzahl HA2 Alarme (48h)", "Zeitstempel letzter HA1 Alarm", "Zeitstempel letzter HA2 Alarm", "Umstallzeit", "Geburtszeit", "Geburtsverlauf"])
    for id in measurements:
        if not measurements[id].did_calve:
            continue

        m = measurements[id]
        #status = "Kuh" if not m.is_heifer else "Faerse"
        status = 0 if m.is_heifer else 1
        geburtsverlauf = verlaufs[id]
        tumstall, tgeburt = None, None
        for e in m.events:
            if e.score == 1:
                tumstall = e.end
                tgeburt = e.real_calving_time

        ha1 = []
        for w in m.ha1_warnings:
            if (tumstall - w).total_seconds() / 3600. > 48.:
                continue
            ha1.append(w)
        last_HA1 = "" if len(ha1) == 0 else str(max(ha1))

        ha2 = []
        for w in m.ha2_warnings:
            if (tumstall - w).total_seconds() / 3600. > 48.:
                continue
            ha2.append(w)
        last_HA2 = "" if len(ha2) == 0 else str(max(ha2))

        f.writerow([str(id), status, len(ha1), len(ha2), last_HA1, last_HA2, str(tumstall), str(tgeburt), geburtsverlauf])

# CSV with all events and stuff
with open("Events.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Typ", "Startzeit", "Endzeit", "Kuh ID", "Status (Kuh / Faerse)"])
    for id in measurements:
        if not measurements[id].did_calve:
            continue

        m = measurements[id]
        status = "Kuh" if not m.is_heifer else "Faerse"
        for e in m.events:
            f.writerow([e.score, str(e.start), str(e.end), str(id), status])

ha1_per_event23 = []
ha2_per_event23 = []
ha1_without_event23 = []
ha2_without_event23 = []
for id in measurements:
    if not measurements[id].did_calve:
        continue

    m = measurements[id]
    has2or3 = len([1 for e in m.events if e.score == 2 or e.score == 3]) > 0

    calving_time = None
    for e in m.events:
        if e.score == 1:
            calving_time = deepcopy(e.end)
    assert calving_time is not None, "lol"

    num_ha1 = 0
    for w in m.ha1_warnings:
        dt = (calving_time - w).total_seconds() / 3600.
        if dt < 48.:
            num_ha1 += 1

    num_ha2 = 0
    for w in m.ha2_warnings:
        dt = (calving_time - w).total_seconds() / 3600.
        if dt < 48.:
            num_ha2 += 1

    if has2or3:
        ha1_per_event23.append(num_ha1)
        ha2_per_event23.append(num_ha2)
    else:
        ha1_without_event23.append(num_ha1)
        ha2_without_event23.append(num_ha2)

ha1_per_event23 = np.array(ha1_per_event23)
ha2_per_event23 = np.array(ha2_per_event23)
ha1_without_event23 = np.array(ha1_without_event23)
ha2_without_event23 = np.array(ha2_without_event23)

print("HA1 with event 2 or 3:                 ", ha1_per_event23.mean())
print("HA2 with event 2 or 3:                 ", ha2_per_event23.mean())
print("HA1 without event 2 or 3:              ", ha1_without_event23.mean())
print("HA2 without event 2 or 3:              ", ha2_without_event23.mean())
print("Number of animals with event 2 or 3:   ", ha1_per_event23.size)
print("Number of animals without event 2 or 3:", ha1_without_event23.size)

# CSV with event 2/3
max_e2 = 6 # maximum number of event 2 within 48h before umstallung
max_e3 = 6
with open("Anzahl_HA1_HA2_Meldungen_wenn_Event_2_3.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    header = ["Kuh ID", "Status (Kuh / Faerse)", "Anzahl Events 2", "Anzahl Events 3", "Anzahl HA1 Meldungen", "Anzahl HA2 Meldungen"]
    for i in range(max_e2):
        header += ["Event 2 #" + str(i+1), "Event 2 Moocall Score #" + str(i+1)]
    for i in range(max_e2):
        header += ["Event 3 #" + str(i+1), "Event 3 Moocall Score #" + str(i+1)]
    for i in range(6):
        header += ["HA1 #" + str(i+1)]
    for i in range(4):
        header += ["HA2 #" + str(i+1)]
    f.writerow(header)

    for id in measurements:
        if not measurements[id].did_calve:
            continue

        m = measurements[id]

        calving_time = None
        for e in m.events:
            if e.score == 1:
                calving_time = deepcopy(e.end)
        assert calving_time is not None, "lol"

        status = "Kuh" if not m.is_heifer else "Faerse"

        e2 = []
        e3 = []
        for e in m.events:
            if (calving_time - e.end).total_seconds() / 3600. > 48.:
                continue
            if e.score == 2:
                e2.append(e.end)
                e2.append(e.moocall_score if e.moocall_score is not None else "")
            elif e.score == 3:
                e3.append(e.end)
                e3.append(e.moocall_score if e.moocall_score is not None else "")

        if len(e2) == 0 and len(e3) == 0:
            continue
        num_e2 = len(e2)
        num_e3 = len(e3)

        while len(e2) < 2*max_e2:
            e2.append("")
        while len(e3) < 2*max_e3:
            e3.append("")

        ha1 = []
        for w in m.ha1_warnings:
            dt = (calving_time - w).total_seconds() / 3600.
            if dt < 48.:
                ha1.append(str(w))
        num_ha1 = len(ha1)
        while len(ha1) < 6:
            ha1.append("")

        ha2 = []
        for w in m.ha2_warnings:
            dt = (calving_time - w).total_seconds() / 3600.
            if dt < 48.:
                ha2.append(str(w))
        num_ha2 = len(ha2)
        while len(ha2) < 6:
            ha2.append("")

        f.writerow([id, status, num_e2, num_e3, num_ha1, num_ha2] + e2 + e3 + ha1 + ha2)

with open("./Events_2_und_3.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Typ (Event 2 / Event 3)", "Kuh ID", "Status (Kuh / Faerse)", "Zeitstempel", "Moocall Score"])

    for id in measurements:
        if not measurements[id].did_calve:
            continue

        m = measurements[id]
        calving_time = None
        for e in m.events:
            if e.score == 1:
                calving_time = deepcopy(e.end)
        assert calving_time is not None, "lol"

        status = "Kuh" if not m.is_heifer else "Faerse"

        for e in m.events:
            if (calving_time - e.end).total_seconds() / 3600. > 48.:
                continue
            if e.score != 2 and e.score != 3:
                continue
            f.writerow([e.score, id, status, e.end] + [e.moocall_score if e.moocall_score is not None else ""])




# Geburtsverlauf nach Kuh und Färse
with open("./Voss-Moocall-Geburtsverlauf_Korrelation_Dystokie_24-Jul-2018.csv") as fh:
    data = list(csv.reader(fh, delimiter=";"))

with open("./Voss-Moocall-Geburtsverlauf_Korrelation_Dystokie_Kuh_Faerse_24-Jul-2018.csv", "w") as fh:
    writer = csv.writer(fh, delimiter=";")
    writer.writerow(data[0] + ["Status (Kuh / Faerse)"])
    for d in data[1:]:
        animal = int(d[0])
        for id in measurements:
            if animal != id:
                continue
            assert measurements[id].did_calve, "What!?"
            writer.writerow(d + ["Faerse" if measurements[id].is_heifer else "Kuh"])

# EVERYTHING
with open("HA1-HA2-Warnungen_alle.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Typ (HA1/HA2)", "Zeitstempel"])
    for id in measurements:
        m = measurements[id]
        for w in m.ha1_warnings:
            f.writerow([id, "HA1", w])
        for w in m.ha2_warnings:
            f.writerow([id, "HA2", w])

with open("Events_alle.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Event-Score", "Moocall-Score", "Zeitstempel", "Sensor-On-Tail", "Sensor-Off-Tail"])
    for id in measurements:
        m = measurements[id]
        for e in m.events:
            f.writerow([id, e.score, e.moocall_score, e.end, e.start, e.sensor_off_tail])

with open("Kuehe_alle.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Kuh ID", "Kuh / Faerse", "Kalbung (0=nein, 1=ja)", "Geburtsverlauf", "Umstallzeit", "Geburtszeit", "Anzahl HA1 Warnungen", "Anzahl HA2 Warnungen", "Anzahl Events"])
    for id in measurements:
        m = measurements[id]
        status = "Faerse" if m.is_heifer else "Kuh"
        calved = 1 if m.did_calve else 0
        geburtsverlauf = "" if 0 == calved else verlaufs[id]
        tumstall, tgeburt = "", ""
        for e in m.events:
            if e.score == 1:
                tumstall = e.end
                tgeburt = e.real_calving_time
        f.writerow([id, status, calved, geburtsverlauf, tumstall, tgeburt, len(m.ha1_warnings), len(m.ha2_warnings), len(m.events)])

# Events pro Tier, OP, Episode (nur Kalbungen)
devents = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
devents_cows = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
devents_heifers = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
episodes_cows = 0
episodes_heifers = 0
op_cows = 0
op_heifers = 0
ncows = 0
nheifers = 0
for id in measurements:
    if not measurements[id].did_calve:
        continue
    m = measurements[id]

    if m.is_heifer:
        nheifers += 1
        op_heifers += m.carry_time
        episodes_heifers += len(m.events)
    else:
        ncows += 1
        op_cows += m.carry_time
        episodes_cows += len(m.events)

    for e in m.events:
        devents[e.score] += 1
        if m.is_heifer:
            devents_heifers[e.score] += 1
        else:
            devents_cows[e.score] += 1

print("Events Calvings ---")
print("Events (total):", devents)
print("Events (cows):", devents_cows)
print("Events (heifers):", devents_heifers)

with open("Events_kalbender_Tiere_pro_Tier_RP_und_Episode.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Score", "Gesamtanzahl", "Anzahl Kuehe", "Anzahl Faersen",
                "Pro Tier", "Pro Kuh", "Pro Faerse",
                "Pro RP [1/h]", "Pro RP (Kuh) [1/h]", "Pro RP (Faerse) [1/h]",
                "Pro Episode", "Pro Episode (Kuh)", "Pro Episode (Faerse)"])

    for e in devents:
        f.writerow([e, devents[e], devents_cows[e], devents_heifers[e],
                    devents[e] / (ncows + nheifers), devents_cows[e] / ncows, devents_heifers[e] / nheifers,
                    devents[e] / (op_cows + op_heifers), devents_cows[e] / op_cows, devents_heifers[e] / op_heifers,
                    devents[e] / (episodes_cows + episodes_heifers), devents_cows[e] / episodes_cows, devents_heifers[e] / episodes_heifers])

print("#episodes (total):  ", episodes_cows + episodes_heifers)
print("#episodes (cows):   ", episodes_cows)
print("#episodes (heifers):", episodes_heifers)
print("rp (total):         ", op_cows + op_heifers)
print("rp (cows):          ", op_cows)
print("rp (heifers):       ", op_heifers)
print("#animals:           ", ncows + nheifers)
print("#cows:              ", ncows)
print("#heifers:           ", nheifers)

# Events pro Tier, OP, Episode (alle Tiere)
devents = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
devents_cows = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
devents_heifers = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }
episodes_cows = 0
episodes_heifers = 0
op_cows = 0
op_heifers = 0
ncows = 0
nheifers = 0
for id in measurements:
    if 8 in [e.score for e in measurements[id].events]:
        print(id)
        continue
    m = measurements[id]

    if m.is_heifer:
        nheifers += 1
        op_heifers += m.carry_time
        episodes_heifers += len(m.events)
    else:
        ncows += 1
        op_cows += m.carry_time
        episodes_cows += len(m.events)

    for e in m.events:
        devents[e.score] += 1
        if m.is_heifer:
            devents_heifers[e.score] += 1
        else:
            devents_cows[e.score] += 1

print("Events All Animals ---")
print("Events (total):", devents)
print("Events (cows):", devents_cows)
print("Events (heifers):", devents_heifers)

with open("Events_aller_Tiere_pro_Tier_RP_und_Episode.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Score", "Gesamtanzahl", "Anzahl Kuehe", "Anzahl Faersen",
                "Pro Tier", "Pro Kuh", "Pro Faerse",
                "Pro RP [1/h]", "Pro RP (Kuh) [1/h]", "Pro RP (Faerse) [1/h]",
                "Pro Episode", "Pro Episode (Kuh)", "Pro Episode (Faerse)"])

    sea = 0.
    sec = 0.
    seh = 0.
    for e in devents:
        ta = str(devents[e] / (ncows + nheifers)).replace(".", ",")
        tc = str(devents_cows[e] / ncows).replace(".", ",")
        th = str(devents_heifers[e] / nheifers).replace(".", ",")
        oa = str(devents[e] / (op_cows + op_heifers)).replace(".", ",")
        oc = str(devents_cows[e] / op_cows).replace(".", ",")
        oh = str(devents_heifers[e] / op_heifers).replace(".", ",")
        ea = str(devents[e] / (episodes_cows + episodes_heifers)).replace(".", ",")
        ec = str(devents_cows[e] / episodes_cows).replace(".", ",")
        eh = str(devents_heifers[e] / episodes_heifers).replace(".", ",")

        sea += devents[e] / (episodes_cows + episodes_heifers)
        sec += devents_cows[e] / episodes_cows
        seh += devents_heifers[e] / episodes_heifers

        f.writerow([e, devents[e], devents_cows[e], devents_heifers[e],
                    ta, tc, th,
                    oa, oc, oh,
                    ea, ec, eh])

print("#episodes (total):  ", episodes_cows + episodes_heifers)
print("#episodes (cows):   ", episodes_cows)
print("#episodes (heifers):", episodes_heifers)
print("rp (total):         ", op_cows + op_heifers)
print("rp (cows):          ", op_cows)
print("rp (heifers):       ", op_heifers)
print("#animals:           ", ncows + nheifers)
print("#cows:              ", ncows)
print("#heifers:           ", nheifers)
print("sum episodes all:   ", sea)
print("sum episodes cow:   ", sec)
print("sum episodes hei:   ", seh)

# IDs and stuff
with open("./ROHDATEN-sortiert-26-Jul-2018.csv") as fh:
    iddata = list(csv.reader(fh, delimiter=","))[1:]

ids = set()
for d in iddata:
    ids.add(int(d[0]))

evalids = [id for id in measurements]
for id in ids:
    if id not in evalids:
        print("Not in evaluation:", id)

# At the end, all the time
#with open("./Geburtsverlauf-23-Jul-2018.csv") as fh:
#    geburtsverlauf = list(csv.reader(fh, delimiter=","))
#
#for id in measurements:
#    if not measurements[id].did_calve:
#        continue
#    for e in measurements[id].events:
#        if e.score == 1:
#            for g in geburtsverlauf:
#                if str(id) == g[0]:
#                    g[2] = str(e.real_calving_time)
#
#with open("./Geburtsverlauf-23-Jul-2018_mit_Abkalbezeit.csv", "w") as fh:
#    writer = csv.writer(fh, delimiter=",")
#    for g in geburtsverlauf:
#        writer.writerow(g)
