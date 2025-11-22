from datetime import datetime
from threading import Timer
from random import randint


class Periodic_Timer_Thread(object):
    def __init__(self, interval, function,
                 comment: str = '', *args: object, **kwargs: object):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.comment = comment
        self.is_running = False
        # print('PTT ',interval)
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(self.comment, *self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            if isinstance(self.interval, list):
                nwi = randint(self.interval[0], self.interval[1])
            elif self.interval < 60:
                nwi = self.interval
            else:
                second = 0
                if self.interval > 1.0:
                    now = datetime.now()
                    second = now.second
                    minute = now.minute
                    hour = now.hour
                    if self.interval <= 60:
                        second = second % self.interval
                    elif self.interval <= 3600:
                        second = (minute*60 + second) % self.interval
                    else:
                        second = (hour*3600 + minute*60 + second) % self.interval
                nwi = self.interval - second
            self._timer = Timer(nwi, self._run)
            self._timer.name = self.comment
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

