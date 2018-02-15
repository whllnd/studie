import csv
import datetime
import numpy as np
import pprint
from copy import deepcopy

class event:
    def __init__(self): # , start_time, end_time, score):
        self.start = None
        self.end = None
        self.score = None
        self.real_calving_time = None

class measurement:

    def __init__(self, cow_id, study):
        self.id = cow_id
        self.study = study
        self.events = []
        self.ha1_warnings = []
        self.ha2_warnings = []
        self.carry_time = 0
        self.did_calve = False

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
        #for w in m.ha1_warnings:
        #    sw)
        #print("Warnings (ha2):")
        #for w in m.ha2_warnings:
        #    print("   ", w)


def parse_time(time):
    try:
        dt = datetime.datetime.strptime(time, "%d.%m.%Y %H:%M")
        return dt
    except ValueError:
        print("Failure:", time)
        raise

def true_positives(measurements, lower, upper, study): # lower in hours, upper in minutes ...

    total_tp_ha1 = 0
    total_tp_ha2 = 0
    for id in measurements:
        m = measurements[id]

        if m.study != study:
            continue

        if not m.did_calve:
            continue

        calving_time = None
        for e in m.events:
            if e.score == 1:
                calving_time = deepcopy(e.end)
        assert calving_time is not None, "Calving time is none?"
        start = calving_time - datetime.timedelta(hours=lower)
        calving_time -= datetime.timedelta(minutes=upper)

        tp_ha1 = sum([1 for w in m.ha1_warnings if w <= calving_time and w >= start])
        tp_ha2 = sum([1 for w in m.ha2_warnings if w <= calving_time and w >= start])

        #if tp_ha1 > 1:
        #    print("Should not have happened (ha1):", id, tp_ha1)
        #    print("start:", start_ha1)
        #    print("calving:", calving_time)
        #    print(m.ha1_warnings)

        #if tp_ha2 > 1:
        #    print("Should not have happened (ha2):", id, tp_ha2)
        #    print("start:", start_ha2)
        #    print("calving:", calving_time)
        #    print(m.ha2_warnings)

        total_tp_ha1 += 1 if tp_ha1 > 0 else 0
        total_tp_ha2 += 1 if tp_ha2 > 0 else 0

    return total_tp_ha1, total_tp_ha2

def false_positives(measurements, lower, upper, study): # lower in hours, upper in minutes ...

    total_fp_ha1 = 0
    total_fp_ha2 = 0
    total_hours = 0
    for id in measurements:
        m = measurements[id]

        if m.study != study:
            continue

        # TODO: This ignores warnings coming after calving ...
        end = deepcopy(m.events[-1].end)
        for e in m.events:
            if e.end > end:
                end = deepcopy(e.end)
        if m.did_calve:
            end = e.end - datetime.timedelta(hours=lower) # Need to shift end to lower threshold

        for e in m.events:
            if e.start > end: # TODO: Exclude those ...
                continue
            event_end = deepcopy(e.end)
            if event_end > end:
                event_end = deepcopy(end)
            if e.start > event_end:
                print("What!?")
                print(e.start)
                print(event_end)

            total_fp_ha1 += sum([1 for w in m.ha1_warnings if w <= event_end and w >= e.start])
            total_fp_ha2 += sum([1 for w in m.ha2_warnings if w <= event_end and w >= e.start])
            total_hours += (event_end - e.start).total_seconds() / 3600

    return total_fp_ha1, total_fp_ha2, int(total_hours)

def true_negatives(measurements, lower, upper, study): # lower in hours, upper in minutes ... TODO: Does it make sense to consider threshs

    total_tn_ha1 = 0
    total_tn_ha2 = 0
    total_hours = 0
    for id in measurements:
        m = measurements[id]

        if m.study != study:
            continue

        # Get global end time
        end = deepcopy(m.events[-1].end)
        for e in m.events:
            if e.end > end:
                end = deepcopy(e.end)
        if m.did_calve:
            end -= datetime.timedelta(hours=lower)

        for e in m.events:
            if e.start > end: # TODO: Exclude those ...
                continue
            event_end = deepcopy(e.end)
            if event_end > end:
                event_end = deepcopy(end)

            # This assumes one warning per hour at max
            hours = int((event_end - e.start).total_seconds() / 3600)
            total_tn_ha1 += hours - sum([1 for w in m.ha1_warnings if w <= event_end and w >= e.start])
            total_tn_ha2 += hours - sum([1 for w in m.ha2_warnings if w <= event_end and w >= e.start])
            total_hours += hours # TODO: How to handle? Should we adapt to thresholds or not?

    return total_tn_ha1, total_tn_ha2, total_hours

def csv_export(filename, dict_list):

    tables_ha1 = []
    tables_ha2 = []
    for i in range(len(dict_list)):
        keys = list(dict_list[i].keys())
        r = len(dict_list[i][keys[0]]) + 1
        mat1,mat2 = np.zeros((r,13)),np.zeros((r,13))
        mat1[1][0],mat2[1][0] = 1, 1
        mat1[2][0],mat2[2][0] = 30, 30
        for j in range(13):
            mat1[0][j] = j
            mat2[0][j] = j

        for k in keys:
            u,l = k[0],k[1]
            if u != 1:
                u = 2
            mat1[u][l] = dict_list[i][k][0]
            mat2[u][l] = dict_list[i][k][1]
            if r == 4:
                mat1[3][l] = dict_list[i][k][2]
                mat2[3][l] = dict_list[i][k][2]

        tables_ha1.append(deepcopy(mat1))
        tables_ha2.append(deepcopy(mat2))
    assert len(tables_ha1) == 2 and len(tables_ha2) == 2, "Ja, was denn?"

    total_ha1 = deepcopy(tables_ha1[0])
    total_ha2 = deepcopy(tables_ha2[0])
    total_ha1[1:,1:] += tables_ha1[1][1:,1:]
    total_ha2[1:,1:] += tables_ha2[1][1:,1:]

    with open(filename, "w") as fh:
        writer = csv.writer(fh, delimiter=";")

        for c in [["Total:", total_ha1, total_ha2], ["Studie 1:", tables_ha1[0], tables_ha2[0]], ["Studie 2:", tables_ha1[1], tables_ha2[1]]]:

            writer.writerow([c[0]])
            tha1, tha2 = c[1], c[2]
            writer.writerow(["HA1"] + [str(int(tha1[0][i])) for i in range(1,13)])
            writer.writerow([str(int(tha1[1][i])) for i in range(13)])
            writer.writerow([str(int(tha1[2][i])) for i in range(13)])
            if r == 4:
                writer.writerow(["Total hours"] + [str(tha1[3][i]) for i in range(1,13)])
            writer.writerow([""])
            writer.writerow(["HA2"] + [str(int(total_ha2[0][i])) for i in range(1,13)])
            writer.writerow([str(int(tha2[1][i])) for i in range(13)])
            writer.writerow([str(int(tha2[2][i])) for i in range(13)])
            if r == 4:
                writer.writerow(["Total hours"] + [str(tha2[3][i]) for i in range(1,13)])
            writer.writerow([""])


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

        if e.end is None or e.start is None:
            print("Both event times are None for id", id)
            raise

        e.score = score
        if score == 1 and len(line[I]) > 0:
            e.real_calving_time = parse_time(line[I])
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
with open("./Moocall_Daten_korrigiert-08-Feb-2018.csv") as fh:
    study = list(csv.reader(fh, delimiter=";"))

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
calvings = [0, 0]
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

    calvings[m.study-1] += c
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


# Evaluation

# 1.) How many cows were involved
print("01.) How many cows were involved in the stuy (total):   ", len(unique_id[0]) + len(unique_id[1]))
print("                                             (studie 1):", len(unique_id[0]))
print("                                             (studie 2):", len(unique_id[1]))

# 2.) How many cows that were involved had calvings
print("02.) How many cows actually calved (total):   ", sum(calvings))
print("                                   (studie 1):", calvings[0])
print("                                   (studie 2):", calvings[1])

# 3.) Average carrying time for calving cows
act1 = sum([measurements[id].carry_time for id in measurements if measurements[id].study == 1 and measurements[id].did_calve]) / calvings[0]
act2 = sum([measurements[id].carry_time for id in measurements if measurements[id].study == 2 and measurements[id].did_calve]) / calvings[1]
act = sum([measurements[id].carry_time for id in measurements if measurements[id].did_calve]) / sum(calvings)
print("03.) Average device carrying time in hours (total):   ", act)
print("                                           (studie 1):", act1)
print("                                           (studie 2):", act2)

# 4.) Average time between F and I
f_to_i = [0,0]
for id in measurements:
    for e in measurements[id].events:
        if e.real_calving_time is not None:
            dt = (e.real_calving_time - e.end).total_seconds() / 3600
            #if dt > 5:
            #    print("Big dt:", id, dt)
            #    print("    end time: ", e.end)
            #    print("    real time:", e.real_calving_time)
            if dt < 0:
                print("    Negative dt for id", id, "dt:", dt)
                print("        end time: ", e.end)
                print("        real time:", e.real_calving_time)
                continue # TODO
            f_to_i[measurements[id].study-1] += dt
print("04.) Average time between F and I (total):   ", sum(f_to_i) / sum(calvings))
print("                                  (studie 1):", f_to_i[0] / calvings[0])
print("                                  (studie 2):", f_to_i[1] / calvings[1])

# 5./6./7.) True positives, false positives, false negatives (TODO: fn = calvings - tp??)
tp = [{}, {}]
fp = [{}, {}]
fn = [{}, {}]
tn = [{}, {}]
for study in range(2):
    for upper in [1, 30]:
        for lower in range(1,13):
            tpos = true_positives(measurements, lower, upper, study + 1)
            fpos = false_positives(measurements, lower, upper, study + 1)
            tp[study][(upper, lower)] = tpos
            fp[study][(upper, lower)] = fpos
            fn[study][(upper, lower)] = [calvings[study] - tpos[0], calvings[study] - tpos[1]]
            tn[study][(upper, lower)] = [fpos[2] - fpos[0], fpos[2] - fpos[1], fpos[2]]
csv_export("true_positives.csv", tp)
csv_export("false_positives.csv", fp)
csv_export("true_negatives.csv", tn)
csv_export("false_negatives.csv", fn)
print("05.) See exported csv files \"true_positives.csv\", \"false_positives.csv\", \"true_negatives.csv\", \"false_negatives.csv\"")

# 11.) Number of different events
noe1 = { 1:0,2:0,3:0,4:0,5:0,7:sevens1,8:0,9:0 }
noe2 = { 1:0,2:0,3:0,4:0,5:0,7:sevens2,8:0,9:0 }
for id in measurements:
    for e in measurements[id].events:
        if e.score not in noe1:
            print("This key is not there!?", e.score)
            continue
        if e.start < datetime.datetime(2017,12,10):
            noe1[e.score] += 1
        else:
            noe2[e.score] += 1
print("11.) Number of different events (total = studie1 + studie2):")
for score in noe1:
    print("    ", score, "->", noe1[score] + noe2[score], "=", noe1[score], "+", noe2[score])

# 12.) Average time between warnings and F and I, respectively
avg_time_ha1 = [0,0]
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
            avg_time_ha1[m.study-1] += dt
            count_ha1[m.study-1] += 1

    for w in m.ha2_warnings:
        if w >= start and w <= calving_time:
            dt = (calving_time - w).total_seconds() / 3600
            #if dt < 0:
            #    print("Negative time difference for id", id, dt)
            #    continue
            #print("ha2:", dt, id)
            avg_time_ha2[m.study-1] += dt
            count_ha2[m.study-1] += 1
print("12.) Average time between warnings and calving in hours (total):   ", sum(avg_time_ha1) / sum(count_ha1), "(ha1)", sum(avg_time_ha2) / sum(count_ha2), "(ha2)")
print("                                                        (studie 1):", avg_time_ha1[0] / count_ha1[0], "(ha1)", avg_time_ha2[0] / count_ha2[0], "(ha2)")
print("                                                        (studie 2):", avg_time_ha1[1] / count_ha1[1], "(ha1)", avg_time_ha2[1] / count_ha2[1], "(ha2)")

# 13.) Monitoring period
mp1 = (e1-s1).total_seconds() / 3600
mp2 = (e2-s2).total_seconds() / 3600
print("13.) Monitoring period in hours (total):   ", mp1 + mp2)
print("                                (studie 1):", mp1)
print("                                (studie 2):", mp2)

