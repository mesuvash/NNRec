import ciso8601
import datetime
import envoy
from datetime import timedelta


def parseDateTime(datetimestring):
    return ciso8601.parse_datetime(datetimestring)


def getDaysSinceX(inputdata, reference=None):
    if reference is None:
        reference = datetime.datetime.now()
    input_time = parseDateTime(inputdata)
    return (reference - input_time).total_seconds() / (86400.0)


def testStartDate(path, days=1):
    r = envoy.run("tail -1 {}".format(path))
    date = r.std_out.partition(" ")[0].strip().split("\t")[-1]
    purchased_time = parseDateTime(date)
    return purchased_time + timedelta(days=days)


def getepochs(datetimestring):
    dt = parseDateTime(datetimestring)
    return int(dt.strftime("%s"))
