import csv
import datetime
import numpy as np

def parse_time(time):
    try:
        dtime = datetime.datetime.strptime(time, "%d.%m.%Y %H:%M")
        return dtime
    except ValueError:
        print("Failure:", time)
        raise

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
animals = {}
with open("./Rohdaten_2018-09-04.csv") as fh:
    f = list(csv.reader(fh, delimiter=";"))
    for d in f:
        id = int(d[0])
        if not id in animals:
            a = animal()
            a.is_heifer = False if d[1] == "Kuh" else True
            animals.calved = True if d[2] == "ja" else False
            animals[id] = a
        if d[5] == "ja": # Event

