import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

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
    # an animals maximum uncertainty is < number of events * 2h
    for id in animals:
        for e in animals[id].events:
            assert e.event_time > e.sont
            animals[id].device_duration += (e.event_time - e.sont).total_seconds() / 3600.

    return animals


# Main #########################################################################

animals = load_data("./Rohdaten_2018-09-05.csv")
switch = datetime.datetime(2017, 11, 3, hour=16)

ev1 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # before switch
ev1c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev1h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev2 = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}  # after switch
ev2c = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
ev2h = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

ev1_dur, ev2_dur = 0, 0
for id in animals:
    a = animals[id]
    for e in a.events:
        if e.event_time <= switch:
            ev1[e.score] += 1
            if a.is_heifer:
                ev1c[e.score] += 1
            else:
                ev1h[e.score] += 1
            ev1_dur += (e.event_time - e.sont).total_seconds() / 3600.
        else:
            ev2[e.score] += 1
            if a.is_heifer:
                ev2c[e.score] += 1
            else:
                ev2h[e.score] += 1
            ev2_dur += (e.event_time - e.sont).total_seconds() / 3600.

with open("Gummihalterung_voher_nacher.csv", "w") as fh:
    f = csv.writer(fh, delimiter=";")
    f.writerow(["Event Score", "Anzahl zuvor", "Anzahl danach", "Anzahl pro Stunde", "Anzahl pro Stunde"])

    for score in ev1:
        f.writerow([score, ev1[score], ev2[score], str(ev1[score] / ev1_dur).replace(".", ","), str(ev2[score] / ev2_dur).replace(".", ",")])
    f.writerow([""])
    f.writerow(["Tragezeitraum vor Wechsel [h]:", str(ev1_dur).replace(".", ",")])
    f.writerow(["Tragezeitraum nach Wechsel [h]:", str(ev2_dur).replace(".", ",")])

