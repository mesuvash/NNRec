import envoy
import progressbar
import scipy.sparse


class Data(object):

    def __init__(self):
        self.users = {}
        self.items = {}
        self.nusers = 0
        self.nitems = 0
        self.include_time = False

    def update_user_item(self, user, item):
        if user not in self.users:
            self.users[user] = self.nusers
            self.nusers += 1
        if item not in self.items:
            self.items[item] = self.nitems
            self.nitems += 1

    def import_ratings(self, filename, shape=None):
        r = envoy.run('wc -l {}'.format(filename))
        num_lines = int(r.std_out.strip().partition(' ')[0])
        bar = progressbar.ProgressBar(maxval=num_lines, widgets=["Loading ratings: ",
                                                                 progressbar.Bar(
                                                                     '=', '[', ']'),
                                                                 ' ', progressbar.Percentage(),

                                                                 ' ', progressbar.ETA()]).start()
        I, J, V = [], [], []
        with open(filename) as f:
            for i, line in enumerate(f):
                if (i % 1000) == 0:
                    bar.update(i % bar.maxval)
                userid, itemid, rating = line.split()
                self.update_user_item(userid, itemid)
                uid = self.users[userid]
                iid = self.items[itemid]
                I.append(uid)
                J.append(iid)
                V.append(float(rating))
        bar.finish()
        if shape is not None:
            _shape = (self.nusers if shape[0] is None else shape[0],
                      self.nitems if shape[1] is None else shape[1])
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=_shape)
        else:
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=(self.nusers, self.nitems))
        self.R = R.tocsr()


def loadTestData(d, testpath):
    r = envoy.run('wc -l {}'.format(testpath))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines, widgets=['Loading test ratings: ',
                                                             progressbar.Bar(
                                                                 '=', '[', ']'),
                                                             ' ', progressbar.Percentage(),

                                                             ' ', progressbar.ETA()]).start()
    users = set(d.users.keys())
    items = set(d.items.keys())
    cold_start_ratings = []

    I, J, V = [], [], []
    with open(testpath) as fp:
        for i, line in enumerate(fp):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)
            user, item, rating = map(
                lambda x: x.lower(), line.strip().split("\t"))
            if user in users and item in items:
                I.append(d.users[user])
                J.append(d.items[item])
                V.append(float(rating))
            else:
                cold_start_ratings.append(float(rating))
    bar.finish()
    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(len(d.users), len(d.items)))
    return R.tocsr(), cold_start_ratings


def loadColdStartTestData(d, testpath):
    users = set(d.users.keys())
    items = set(d.items.keys())
    cold_start_ratings = []
    with open(testpath) as fp:
        for i, line in enumerate(fp):
            user, item, rating = map(
                lambda x: x.lower(), line.strip().split("\t"))
            if (user not in users) or (item not in items):
                cold_start_ratings.append(float(rating))
    return cold_start_ratings


def loadTrainTest(train_path, test_path, shape=None):
    d = Data()
    d.import_ratings(train_path, shape)
    test, cold = loadTestData(d, test_path)
    train = d.R.copy()
    return train, test, cold
