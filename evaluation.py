import csv
import datetime
import numpy as np

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
        self.calving_time = None
        self.rehousing_time = None

# Read input file and store data in dictionary
def load_data(fname):

    A,B,C,D,E,F,G,H,I,J,K,L,M,N=0,1,2,3,4,5,6,7,8,9,10,11,12,13
    animals = {}
    with open(fname) as fh:

        f = list(csv.reader(fh, delimiter=";"))[1:]
        for d in f:

            id = int(d[A])
            if not id in animals:
                a = animal()
                a.is_heifer = False if d[B] == "Kuh" else True
                a.calved = True if d[C] == "ja" else False
                animals[id] = a

            if d[F] == "ja": # Event
                e = event()
                e.score = int(d[G])
                e.moocall_score = -1 if len(d[K]) == 0 else int(d[K])
                e.event_time = parse_time(d[I])
                e.sont = parse_time(d[H])
                e.sofft = None if len(d[J]) == 0 else parse_time(d[J])
                animals[id].events.append(e)

            if d[L] == "ja": # Alarm
                if d[M] == "HA1h":
                    animals[id].ha1_alarms.append(parse_time(d[N]))
                else:
                    animals[id].ha2_alarms.append(parse_time(d[N]))

    return animals


# Main #########################################################################

animals = load_data("./Rohdaten_2018-09-04.csv")

