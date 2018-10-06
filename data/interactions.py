import datetime

class Interaction:
    def __init__(self,uid,pid,ts,fold='train'):
        self.uid = uid
        self.pid = pid
        self.ts = ts
        self.dt = datetime.datetime.fromtimestamp(ts)
        self.fold = fold

class InteractionSequence:
    def __init__(self,uid,interactions):
        self.uid = uid
        self.interactions = sorted(interactions,key=lambda x : x.dt)

    def get_max_dt(self,fold=None):
        if fold:
            return [intc.dt for intc in self.interactions if intc.fold == fold][-1]
        else :
            return [intc.dt for intc in self.interactions][-1]

    def get_min_dt(self, fold=None):
        if fold:
            return [intc.dt for intc in self.interactions if intc.fold == fold][0]
        else:
            return [intc.dt for intc in self.interactions][0]

    def keep_items(self,pids):
        self.interactions = [intc for intc in self.interactions if intc.pid in pids]

    def remove_items(self,pids):
        self.interactions = [intc for intc in self.interactions if intc.pid not in pids]

    def remove_before(self,dt):
        self.interactions = [intc for intc in self.interactions if intc.dt > dt]

    def remove_after(self,dt):
        self.interactions = [intc for intc in self.interactions if intc.dt < dt]

    def __len__(self):
        return len(self.interactions)

    def __iter__(self):
        return iter(self.interactions)

    def __getitem__(self, index):
        return self.interactions[index]

    def __setitem__(self, index, value):
        self.interactions[index] = value
