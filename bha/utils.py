import time
import datetime

def counter(iterable, countevery=100, total=None, noun="item", verb="completed"):
    """Logs a status and timing update every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count%countevery:
            current_time = time.time()
            rate = float(count + 1) / (current_time - start_time)

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f %ss/second"%(rate, noun)
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/%s"%((rate ** -1), noun)

            if total is not None:
                remitems = total - (count + 1)
                remtime = remitems / rate
                #timestr = ", %s remaining" % time.strftime('%j:%H:%M:%S', time.gmtime(remtime))
                timestr = ", %s remaining" % str(datetime.timedelta(seconds=remtime)).split(".", 2)[0]
                itemstr = "%d/%d" % (count + 1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count + 1)

            formatted_str = "%s %ss %s (%s%s)" % (itemstr, noun, verb, ratestr, timestr)
            print(formatted_str)
