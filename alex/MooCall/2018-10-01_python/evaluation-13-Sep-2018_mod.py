import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pprint

# Convert time strings to datetime objects
def parse_time(time):
    try:
        dtime = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        return dtime
    except ValueError:
        print("Datetime conversion failed for string:", time)
        raise

# A few simple classes to cover our needs
class event:
    def __init__(self):
        self.score = None
        self.event_time = None
        self.sont = None
        self.sofft = None
        self.moocall_score = None

    def __lt__(self, other):
        return self.start < other.start

class animal:

    def __init__(self):
        self.id = None
        self.events = []
        self.ha1_alarms = []
        self.ha2_alarms = []
        self.device_duration = 0
        self.calved = False
        self.is_heifer = False
        self.rehousing_time = None
        self.calving_time = None
        self.calving_condition = None

# Read input file and store data in dictionary
def load_data(fname):

    A,B,C,D,E,F,G,H,I,J,K,L,M,N,O=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
    animals = {}
    with open(fname) as fh:

        f = list(csv.reader(fh, delimiter=";"))[1:]
        for d in f:

            id = int(d[A])
            if not id in animals:
                a = animal()
                a.is_heifer = False if d[B] == "Kuh" else True
                a.calved = True if d[C] == "ja" else False
                a.calving_time = parse_time(d[E])
                if a.calved:

                    # There are 4 cases (ids: 83, 1164, 1231, 369) in which
                    # stage II (or rehousing) "comes after" the actual birth
                    assert len(d[D]) > 0
                    assert len(d[F]) > 0
                    a.rehousing_time = parse_time(d[D])
                    a.calving_condition = int(d[F])
                animals[id] = a

            if d[G] == "ja": # Event
                e = event()
                e.score = int(d[H])
                e.moocall_score = -1 if len(d[L]) == 0 else int(d[L])
                e.event_time = parse_time(d[J])
                e.sont = parse_time(d[I])
                e.sofft = None if len(d[K]) == 0 else parse_time(d[K])
                animals[id].events.append(e)

            if d[M] == "ja": # Alarm
                if d[N] == "HA1h":
                    animals[id].ha1_alarms.append(parse_time(d[O]))
                else:
                    animals[id].ha2_alarms.append(parse_time(d[O]))

    # Calculate the time that the device was attached to the animal by
    # taking the sum of each event duration, i.e. the time between
    # event start (sensor on tail, sont) and event end (event time);
    # this can have a maximum uncertainty of two hours per event, so
    # an animals maximum uncertainty is < number of animal's events * 2h
    for id in animals:
        for e in animals[id].events:
            assert e.event_time > e.sont
            animals[id].device_duration += (e.event_time - e.sont).total_seconds() / 3600.

    return animals


# Main #########################################################################

animals = load_data("./2018-09-14_rohdaten_korr.csv")

# Rubber pads
switch = datetime.datetime(2017, 11, 3, hour=16)
ev1 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # before switch
ev1c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev1h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev2 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # after switch
ev2c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev2h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

fev1 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # before switch
fev1c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
fev1h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
fev2 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # after switch
fev2c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
fev2h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

ev1_dur, ev2_dur = 0, 0
score1 = 0
for id in animals:
    a = animals[id]

    first_event = a.events[0]
    if first_event.event_time <= switch:
        fev1[first_event.score] += 1
        if a.is_heifer:
            fev1h[first_event.score] += 1
        else:
            fev1c[first_event.score] += 1
    else:
        fev2[first_event.score] += 1
        if a.is_heifer:
            fev2h[first_event.score] += 1
        else:
            fev2c[first_event.score] += 1

    for e in a.events:
        if e.score == 1:
            score1 += 1
        if e.event_time <= switch:
            ev1[e.score] += 1
            if a.is_heifer:
                ev1h[e.score] += 1
            else:
                ev1c[e.score] += 1
            ev1_dur += (e.event_time - e.sont).total_seconds() / 3600.
        else:
            ev2[e.score] += 1
            if a.is_heifer:
                ev2h[e.score] += 1
            else:
                ev2c[e.score] += 1
            ev2_dur += (e.event_time - e.sont).total_seconds() / 3600.

with open("Gummihalterung_vorher_nachher_alle_Events.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Score", "Anzahl zuvor", "Anzahl danach", "Anzahl pro Stunde", "Anzahl pro Stunde"])

    for score in ev1:
        f.writerow([score, ev1[score], ev2[score], str(ev1[score] / ev1_dur).replace(".", ","), str(ev2[score] / ev2_dur).replace(".", ",")])
    f.writerow([""])
    f.writerow(["Tragezeitraum vor Wechsel [h]:", str(ev1_dur).replace(".", ",")])
    f.writerow(["Tragezeitraum nach Wechsel [h]:", str(ev2_dur).replace(".", ",")])

with open("Gummihalterung_vorher_nachher_erste_Events.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Score", "Anzahl erster Events zuvor", "Anzahl erster Events danach", "Anzahl pro Stunde", "Anzahl pro Stunde"])

    for score in fev1:
        f.writerow([score, fev1[score], fev2[score], str(fev1[score] / ev1_dur).replace(".", ","), str(fev2[score] / ev2_dur).replace(".", ",")])

# Number of animals
n_cows = sum([1 for id in animals if not animals[id].is_heifer])
n_heif = sum([1 for id in animals if animals[id].is_heifer])
n_total = len(animals)
assert n_cows + n_heif == n_total

# Number of calvings
n_calvings = sum([1 for id in animals if animals[id].calved])
n_calvings_cows = sum([1 for id in animals if animals[id].calved and not animals[id].is_heifer])
n_calvings_heif = sum([1 for id in animals if animals[id].calved and animals[id].is_heifer])
assert n_calvings_cows + n_calvings_heif == n_calvings

# Events after initial attachment
first_events_total = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
first_events_cows  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
first_events_heif  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}

# Events for all animals
events_total = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
events_cows  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
events_heif  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}

# Events for all calvings
events_calvings_total = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
events_calvings_cows  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}
events_calvings_heif  = {1:0, 2:0, 3:0, 4:0, 5:0, 9:0}

# Event histograms for all calvings
events_hist_total = {1:[], 2:[], 3:[], 4:[], 5:[], 9:[]}
events_hist_cows  = {1:[], 2:[], 3:[], 4:[], 5:[], 9:[]}
events_hist_heif  = {1:[], 2:[], 3:[], 4:[], 5:[], 9:[]}

# Number of Risk Periods [h]
rp = 0
rp_cows = 0
rp_heif = 0
rp_calvings = 0
rp_calvings_cows = 0
rp_calvings_heif = 0

# Number of sensor attachments (= number of events)
attachments = 0
attachments_cows = 0
attachments_heif = 0
attachments_calvings = 0
attachments_calvings_cows = 0
attachments_calvings_heif = 0

# Number of (first/last) HA1h/2h alarms of all animals
n_ha1 = 0
n_ha2 = 0
n_first_ha1 = 0
n_first_ha2 = 0
n_last_ha1 = 0
n_last_ha2 = 0

# True positives for 1h, 2h, 3h prior to stage II, i.e. rehousing (Umstallzeit)
tp_ha1_1h = 0
tp_ha1_2h = 0
tp_ha1_3h = 0
tp_ha2_1h = 0
tp_ha2_2h = 0
tp_ha2_3h = 0

# Time difference between rehousing/stage II and actual birth
dt_stageII_birth = []
dt_stageII_birth_cows = []
dt_stageII_birth_heif = []

# Time difference between (first/last) alarms and stage II (which looks rather ridiculous)
dt_ha1_stageII = []
dt_ha1_stageII_cows = []
dt_ha1_stageII_heif = []
dt_ha2_stageII = []
dt_ha2_stageII_cows = []
dt_ha2_stageII_heif = []
dt_first_ha1_stageII = []
dt_first_ha1_stageII_cows = []
dt_first_ha1_stageII_heif = []
dt_first_ha2_stageII = []
dt_first_ha2_stageII_cows = []
dt_first_ha2_stageII_heif = []
dt_last_ha1_stageII = []
dt_last_ha1_stageII_cows = []
dt_last_ha1_stageII_heif = []
dt_last_ha2_stageII = []
dt_last_ha2_stageII_cows = []
dt_last_ha2_stageII_heif = []

# Gather statistics
for id in animals:
    a = animals[id]

    # Events after initial attachment
    first_events_total[a.events[0].score] += 1
    if a.is_heifer:
        first_events_heif[a.events[0].score] += 1
    else:
        first_events_cows[a.events[0].score] += 1

    # Events for animals
    for e in a.events:
        events_total[e.score] += 1
        if a.is_heifer:
            events_heif[e.score] += 1
        else:
            events_cows[e.score] += 1

    # Events for calvings
    if a.calved:
        for e in a.events:
            events_calvings_total[e.score] += 1
            if a.is_heifer:
                events_calvings_heif[e.score] += 1
            else:
                events_calvings_cows[e.score] += 1

    # Event histograms
    if a.calved:
        for e in a.events:
            dt = (a.rehousing_time - e.event_time).total_seconds() / 3600.
            events_hist_total[e.score].append(dt)
            if a.is_heifer:
                events_hist_heif[e.score].append(dt)
            else:
                events_hist_cows[e.score].append(dt)

    # Risk periods and number of attachments
    rp += a.device_duration
    attachments += len(a.events)
    if a.is_heifer:
        rp_heif += a.device_duration
        attachments_heif += len(a.events)
    else:
        rp_cows += a.device_duration
        attachments_cows += len(a.events)
    if a.calved:
        rp_calvings += a.device_duration
        attachments_calvings += len(a.events)
        if a.is_heifer:
            rp_calvings_heif += a.device_duration
            attachments_calvings_heif += len(a.events)
        else:
            rp_calvings_cows += a.device_duration
            attachments_calvings_cows += len(a.events)

# HERE WE DELETE ANIMALS DUE TO UNUSUAL HIGH ALARM RATES #######################
#for id in [2641, 1899, 3909, 1304]:
#    del animals[id]

for id in animals:
    a = animals[id]

    # Alarms
    if len(a.ha1_alarms) > 0:
        n_first_ha1 += 1
        n_ha1 += len(a.ha1_alarms)
    if len(a.ha2_alarms) > 0:
        n_first_ha2 += 1
        n_ha2 += len(a.ha2_alarms)

    # True positives: calculated by taking the time difference between the first
    # alarm (which is the minimum element of each alarm list) and the rehousing
    # time; then, increase all true positive counter for which the dt is less
    # than the corresponding threshold; note: alarms that came after rehousing
    # are ignored
    if a.calved:
        if len(a.ha1_alarms) > 0:
            dt = (a.rehousing_time - min(a.ha1_alarms)).total_seconds() / 3600.
            if dt >= 0. and dt <= 1.:
                tp_ha1_1h += 1
            if dt >= 0. and dt <= 2.:
                tp_ha1_2h += 1
            if dt >= 0. and dt <= 3.:
                tp_ha1_3h += 1

        if len(a.ha2_alarms) > 0:
            dt = (a.rehousing_time - min(a.ha2_alarms)).total_seconds() / 3600.
            if dt >= 0. and dt <= 1.:
                tp_ha2_1h += 1
            if dt >= 0. and dt <= 2.:
                tp_ha2_2h += 1
            if dt >= 0. and dt <= 3.:
                tp_ha2_3h += 1

    # Time difference stage II and actual birth
    if a.calved:
        dt = (a.calving_time - a.rehousing_time).total_seconds() / 3600.
        if dt >= 0.: # Only consider cases with positive difference
            dt_stageII_birth.append(dt)
            if a.is_heifer:
                dt_stageII_birth_heif.append(dt)
            else:
                dt_stageII_birth_cows.append(dt)

    # Time difference alarms and stage II; again, differences < 0. are ignored;
    # this is exhaustingly tedious and ugly code
    if a.calved:

        min_dt, max_dt = np.inf, -np.inf
        for w in a.ha1_alarms:
            dt = (a.rehousing_time - w).total_seconds() / 3600.
            if dt < 0.:
                continue
            dt_ha1_stageII.append(dt)
            if a.is_heifer:
                dt_ha1_stageII_heif.append(dt)
            else:
                dt_ha1_stageII_cows.append(dt)
            min_dt = min(min_dt, dt)
            max_dt = max(max_dt, dt)

        if min_dt < np.inf:
            dt_first_ha1_stageII.append(max_dt)
            dt_last_ha1_stageII.append(min_dt)
            if a.is_heifer:
                dt_first_ha1_stageII_heif.append(max_dt)
                dt_last_ha1_stageII_heif.append(min_dt)
            else:
                dt_first_ha1_stageII_cows.append(max_dt)
                dt_last_ha1_stageII_cows.append(min_dt)

        min_dt, max_dt = np.inf, -np.inf
        for w in a.ha2_alarms:
            dt = (a.rehousing_time - w).total_seconds() / 3600.
            if dt < 0.:
                continue
            dt_ha2_stageII.append(dt)
            if a.is_heifer:
                dt_ha2_stageII_heif.append(dt)
            else:
                dt_ha2_stageII_cows.append(dt)
            min_dt = min(min_dt, dt)
            max_dt = max(max_dt, dt)

        if min_dt < np.inf:
            dt_first_ha2_stageII.append(max_dt)
            dt_last_ha2_stageII.append(min_dt)
            if a.is_heifer:
                dt_first_ha2_stageII_heif.append(max_dt)
                dt_last_ha2_stageII_heif.append(min_dt)
            else:
                dt_first_ha2_stageII_cows.append(max_dt)
                dt_last_ha2_stageII_cows.append(min_dt)



print("\nBasic statistics:")
print("---------------------------------------\n")
print("Number of animals:", n_total)
print("Number of cows:   ", n_cows)
print("Number of heifers:", n_heif)
print("")
print("Number of calvings:          ", n_calvings)
print("Number of calvings (cows):   ", n_calvings)
print("Number of calvings (heifers):", n_calvings)
print("")
print("Sum device carrying time [h]:          ", np.array([animals[id].device_duration for id in animals]).sum())
print("Sum device carrying time (cows) [h]:   ", np.array([animals[id].device_duration for id in animals if not animals[id].is_heifer]).sum())
print("Sum device carrying time (heifers) [h]:", np.array([animals[id].device_duration for id in animals if animals[id].is_heifer]).sum())
print("")
print("Average device carrying time [h]:          ", np.array([animals[id].device_duration for id in animals]).mean())
print("Average device carrying time (cows) [h]:   ", np.array([animals[id].device_duration for id in animals if not animals[id].is_heifer]).mean())
print("Average device carrying time (heifers) [h]:", np.array([animals[id].device_duration for id in animals if animals[id].is_heifer]).mean())
print("")
print("Average device carrying time when calving [h]:          ", np.array([animals[id].device_duration for id in animals if animals[id].calved]).mean())
print("Average device carrying time when calving (cows) [h]:   ", np.array([animals[id].device_duration for id in animals if not animals[id].is_heifer and animals[id].calved]).mean())
print("Average device carrying time when calving (heifers) [h]:", np.array([animals[id].device_duration for id in animals if animals[id].is_heifer and animals[id].calved]).mean())
print("")
print("Average time between stage II and birth [h]:          ", np.array(dt_stageII_birth).mean())
print("Average time between stage II and birth (cows) [h]:   ", np.array(dt_stageII_birth_cows).mean())
print("Average time between stage II and birth (heifers) [h]:", np.array(dt_stageII_birth_heif).mean())

print("\nTable 1: Event distribution after initial attachment for all animals (180)")
print("--------------------------------------------------------------------------\n")
print("Number of animals:", n_total)
print("Number of cows:   ", n_cows)
print("Number of heifers:", n_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", first_events_total[score], "(", first_events_total[score] / n_total, ")")
    print("\tCows:   ", first_events_cows[score], "(", first_events_cows[score] / n_cows, ")")
    print("\tHeifers:", first_events_heif[score], "(", first_events_heif[score] / n_heif, ")")

print("\nTable 2: Event distribution for calvings only (118):")
print("----------------------------------------------------\n")
print("Number of calvings:          ", n_calvings)
print("Number of calvings (cows):   ", n_calvings_cows)
print("Number of calvings (heifers):", n_calvings_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", events_calvings_total[score], "(", events_calvings_total[score] / n_calvings, ")")
    print("\tCows:   ", events_calvings_cows[score], "(", events_calvings_cows[score] / n_calvings_cows, ")")
    print("\tHeifers:", events_calvings_heif[score], "(", events_calvings_heif[score] / n_calvings_heif, ")")

print("\nTable 3: Event distribution for calvings only regarding number of RPs:")
print("----------------------------------------------------\n")
print("Number of RPs when calving:          ", rp_calvings)
print("Number of RPs when calving (cows):   ", rp_calvings_cows)
print("Number of RPs when calving (heifers):", rp_calvings_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", events_calvings_total[score] / rp_calvings)
    print("\tCows:   ", events_calvings_cows[score] / rp_calvings_cows)
    print("\tHeifers:", events_calvings_heif[score] / rp_calvings_heif)

print("\nTable 4: Event distribution regarding number of RPs for all animals (180):")
print("----------------------------------------------------\n")
print("Number of RPs:          ", rp)
print("Number of RPs (cows):   ", rp_cows)
print("Number of RPs (heifers):", rp_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", events_total[score] / rp)
    print("\tCows:   ", events_cows[score] / rp_cows)
    print("\tHeifers:", events_heif[score] / rp_heif)

print("\nTable 5: Event distribution for calvings only regarding number of attachments:")
print("----------------------------------------------------\n")
print("Number of attachments when calving:          ", attachments_calvings)
print("Number of attachments when calving (cows):   ", attachments_calvings_cows)
print("Number of attachments when calving (heifers):", attachments_calvings_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", events_calvings_total[score] / attachments_calvings)
    print("\tCows:   ", events_calvings_cows[score] /  attachments_calvings_cows)
    print("\tHeifers:", events_calvings_heif[score] /  attachments_calvings_heif)

print("\nTable 6: Event distribution regarding number of attachments for all animals (180):")
print("----------------------------------------------------\n")
print("Number of attachments:          ", attachments)
print("Number of attachments (cows):   ", attachments_cows)
print("Number of attachments (heifers):", attachments_heif)
print("")
for score in first_events_total:
    print("Event", score)
    print("\tTotal:  ", events_total[score] / attachments)
    print("\tCows:   ", events_cows[score] / attachments_cows)
    print("\tHeifers:", events_heif[score] / attachments_heif)

print("\nAlarm sensitivity (according to statistics meeting) for all animals (180):")
print("--------------------------------------------------------------------------\n")
print("Number of HA1h alarms:               ", n_ha1)
print("Number of HA2h alarms:               ", n_ha2)
print("Number of calving:                   ", n_calvings)
print("Ratio of all HA1h alarms to calvings:", n_ha1 / n_calvings)
print("Ratio of all HA2h alarms to calvings:", n_ha2 / n_calvings)
print("Number of all 'first' HA1h alarms:   ", n_first_ha1)
print("Number of all 'first' HA2h alarms:   ", n_first_ha2)
print("")
print("True positives HA1h 1h (abs):", tp_ha1_1h)
print("True positives HA1h 2h (abs):", tp_ha1_2h)
print("True positives HA1h 3h (abs):", tp_ha1_3h)
print("")
print("True positives HA1h 1h (relative to number of first alarms):", tp_ha1_1h / n_first_ha1)
print("True positives HA1h 2h (relative to number of first alarms):", tp_ha1_2h / n_first_ha1)
print("True positives HA1h 3h (relative to number of first alarms):", tp_ha1_3h / n_first_ha1)
print("")
print("True positives HA2h 1h (abs):", tp_ha2_1h)
print("True positives HA2h 2h (abs):", tp_ha2_2h)
print("True positives HA2h 3h (abs):", tp_ha2_3h)
print("")
print("True positives HA2h 1h (relative to number of first alarms):", tp_ha2_1h / n_first_ha2)
print("True positives HA2h 2h (relative to number of first alarms):", tp_ha2_2h / n_first_ha2)
print("True positives HA2h 3h (relative to number of first alarms):", tp_ha2_3h / n_first_ha2)

print("\nTime differences between alarms and rehousing time for calvings only (114):")
print("---------------------------------------------------------------------------\n")
print("Average time between all HA1h and stage II [h]:            ", np.array(dt_ha1_stageII).mean())
print("Average time between all HA1h and stage II (cows) [h]:     ", np.array(dt_ha1_stageII_cows).mean())
print("Average time between all HA1h and stage II (heifers) [h]:  ", np.array(dt_ha1_stageII_heif).mean())
print("Average time between first HA1h and stage II [h]:          ", np.array(dt_first_ha1_stageII).mean())
print("Average time between first HA1h and stage II (cows) [h]:   ", np.array(dt_first_ha1_stageII_cows).mean())
print("Average time between first HA1h and stage II (heifers) [h]:", np.array(dt_first_ha1_stageII_heif).mean())
print("Average time between last HA1h and stage II [h]:           ", np.array(dt_last_ha1_stageII).mean())
print("Average time between last HA1h and stage II (cows) [h]:    ", np.array(dt_last_ha1_stageII_cows).mean())
print("Average time between last HA1h and stage II (heifers) [h]: ", np.array(dt_last_ha1_stageII_heif).mean())

# Histogram
#hist = np.histogram(events_hist_total[2], bins=20)
#for score in events_hist_total:
#    if score == 1: # Skip calving event since it is the reference time
#        continue
#    hist = np.histogram(events_hist_total[score], bins=20)
#    plt.bar(hist[1][:-1], hist[0], width=4, label="Event " + str(score))
#plt.title("Time differences between event and stage II, i.e. rehousing (114 animals)")
#plt.xlabel("dt [h]")
#plt.ylabel("Number of events")
#plt.legend()
#plt.show()

# Extensions
alarms_inside_gaps = []
for id in animals:
    a = animals[id]
    for w in a.ha1_alarms:
        for i in range(1, len(a.events)):
            if w < a.events[i].sont and w > a.events[i-1].event_time:
                alarms_inside_gaps.append([id, "HA1", w, a.events[i-1].event_time, a.events[i].sont, (a.events[i].sont - a.events[i-1].event_time).total_seconds() / 3600., (w - a.events[i-1].event_time).total_seconds() / 3600.])
    for w in a.ha2_alarms:
        for i in range(1, len(a.events)):
            if w < a.events[i].sont and w > a.events[i-1].event_time:
                alarms_inside_gaps.append([id, "HA2", w, a.events[i-1].event_time, a.events[i].sont, (a.events[i].sont - a.events[i-1].event_time).total_seconds() / 3600., (w - a.events[i-1].event_time).total_seconds() / 3600.])
alarms_inside_gaps.sort(key=lambda x: x[-1])

with open("alarms_inside_event_gaps.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["ID", "Alarmtyp (HA1/HA2)", "Alarmzeitpunkt", "Letztes Event", "Naechstes Event (SonT)", "Lueckengroesse [h]", "Zeitunterschied letztem Event und Alarm [h]"])
    for alarm in reversed(alarms_inside_gaps):
        f.writerow(alarm)

# Delete alarms inside gaps from data
for alarm in alarms_inside_gaps:
    id = alarm[0]
    w = alarm[2]
    if alarm[-1] > 1:
        if alarm[1] == "HA1":
            animals[id].ha1_alarms.remove(w)
        else:
            animals[id].ha2_alarms.remove(w)

#print("Events that do not obey order:") # There are none
#for id in animals:
#    a = animals[id]
#    for i in range(1, len(a.events)):
#        if a.events[i-1].event_time > a.events[i].sont:
#            print(id, a.events[i-1].event_time, a.events[i].sont)

all_gaps = []
for thresh in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24]:
    with open("./confusion_table_" + str(thresh) + "h.csv", "w") as fh, open("./confusion_table_" + str(thresh) + "h_ha1.csv", "w") as fh1, open("./confusion_table_" + str(thresh) + "h_ha2.csv", "w") as fh2:
        f = csv.writer(fh, delimiter=";")
        f1 = csv.writer(fh1, delimiter=";")
        f2 = csv.writer(fh2, delimiter=";")
        f.writerow(["ID", "RN", "FN", "RP", "FP", "Sensitivitaet", "Spezifitaet", "PPV", "NPV"])
        f1.writerow(["ID", "RN", "FN", "RP", "FP", "Sensitivitaet", "Spezifitaet", "PPV", "NPV"])
        f2.writerow(["ID", "RN", "FN", "RP", "FP", "Sensitivitaet", "Spezifitaet", "PPV", "NPV"])

        for id in animals:
            a = animals[id]
            if not a.calved:
                continue

            # Determine gaps between events
            gaps = []
            for i in range(1, len(a.events)):
                gaps.append([a.events[i-1].event_time, a.events[i].sont])
                if thresh == 1:
                    all_gaps.append([id, (a.events[i].sont - a.events[i-1].event_time).total_seconds() / 3600.])
            assert len(gaps)+1 == len(a.events)

            # Calculate net time between alarm and rehousing/stage II
            d_ha1 = {}
            for w in a.ha1_alarms:
                dt = (a.rehousing_time - w).total_seconds() / 3600.
                if dt < 0:
                    continue
                for g in reversed(gaps):
                    if w < g[0]:
                        dt -= (g[1]-g[0]).total_seconds() / 3600.
                    elif w < g[1]:
                        dt -= (g[1]-w).total_seconds() / 3600.
                        break
                    else:
                        break
                k = int(dt // thresh) # Determine "bin"
                if k in d_ha1:
                    d_ha1[k] += 1 # Keep actual count
                else:
                    d_ha1[k] = 1

            d_ha2 = {}
            for w in a.ha2_alarms:
                dt = (a.rehousing_time - w).total_seconds() / 3600.
                if dt < 0:
                    continue
                for g in reversed(gaps):
                    if w < g[0]:
                        dt -= (g[1]-g[0]).total_seconds() / 3600.
                    elif w < g[1]:
                        dt -= (g[1]-w).total_seconds() / 3600.
                        break
                    else:
                        break
                k = int(dt // thresh) # Determine "bin"
                if k in d_ha2:
                    d_ha2[k] += 1 # Keep actual count
                else:
                    d_ha2[k] = 1

            d_all = {}
            for k in d_ha1:
                d_all[k] = d_ha1[k]
            for k in d_ha2:
                if k in d_all:
                    d_all[k] += d_ha2[k]
                else:
                    d_all[k] = d_ha2[k]

            # Calculate TP, FP, TN, FN
            nbins = int(np.ceil(a.device_duration / thresh)) # This benefits Moocall
            FP = len([k for k in d_all if k != 0])
            RP = 1 if 0 in d_all else 0
            RN = nbins - 1 - FP
            if RN == -1:
                print("-1 for both")
                print(d_all)
                print(FP)
                print("For Intervall")
                print(thresh)
                raise
            FN = 1 if 0 not in d_all else 0
            if RP + FN > 0:
                se = RP / (RP + FN)
            else:
                se = ""
            if RN + FP > 0:
                sp = RN / (RN + FP)
            else:
                sp = ""
            if RP + FP > 0:
                ppv = RP / (RP + FP)
            else:
                ppv = ""
            if RN + FN > 0:
                npv = RN / (RN + FN)
            else:
                npv = ""

            f.writerow([id, RN, FN, RP, FP, se, sp, ppv, npv])

            # Individual
            FP1 = len([k for k in d_ha1 if k != 0])
            RP1 = 1 if 0 in d_ha1 else 0
            RN1 = nbins - 1 - FP1
            FN1 = 1 if 0 not in d_ha1 else 0
            if RN1 == -1:
                print("-1 for ha1")
                print(d_ha1)
                print(FP1)
                print(thresh)
                raise
            if RP1 + FN1 > 0:
                se1 = RP1 / (RP1 + FN1)
            else:
                se1 = ""
            if RN1 + FP1 > 0:
                sp1 = RN1 / (RN1 + FP1)
            else:
                sp1 = ""
            if RP1 + FP1 > 0:
                ppv1 = RP1 / (RP1 + FP1)
            else:
                ppv1 = ""
            if RN1 + FN1 > 0:
                npv1 = RN1 / (RN1 + FN1)
            else:
                npv1 = ""

            f1.writerow([id, RN1, FN1, RP1, FP1, se1, sp1, ppv1, npv1])

            FP2 = len([k for k in d_ha2 if k != 0])
            RP2 = 1 if 0 in d_ha2 else 0
            RN2 = nbins - 1 - FP2
            FN2 = 1 if 0 not in d_ha2 else 0
            if RN2 == -1:
                print("-1 for ha2")
                print(d_ha2)
                print(FP2)
                print("For Intervall")
                print(thresh)
                raise
            if RP2 + FN2 > 0:
                se2 = RP2 / (RP2 + FN2)
            else:
                se2 = ""
            if RN2 + FP2 > 0:
                sp2 = RN2 / (RN2 + FP2)
            else:
                sp2 = ""
            if RP2 + FP2 > 0:
                ppv2 = RP2 / (RP2 + FP2)
            else:
                ppv2 = ""
            if RN2 + FN2 > 0:
                npv2 = RN2 / (RN2 + FN2)
            else:
                npv2 = ""

            f2.writerow([id, RN2, FN2, RP2, FP2, se2, sp2, ppv2, npv2])

# Look at maximum gaps and where they happen
all_gaps.sort(key=lambda x: x[1])
with open("event_gaps.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["ID", "Lueckengroesse [h]"])
    for gap in reversed(all_gaps):
        f.writerow([gap[0], gap[1]])

