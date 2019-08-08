import time
from qcodes.dataset.measurements import Measurement


class event():
    def __init__(self, time, step, fn_before):
        self.time = time
        self.step = step
        self.fn_before = fn_before


def time_sweep(dependents, event_list, fn_before=None, fn_after=None,
               exp=None, station=None):
    """
    Takes a list of Events which can run some event and then measure
    for a period of time. Eg
    Set vtun to 3v for 5 secs, then bump to 10v for 2 secs, then
    back to 3v for another 5 secs.
    """
    meas = Measurement(exp=exp, station=station)
    meas.register_custom_parameter("time", label="Time", unit="S")

    if callable(fn_before):
        meas.add_before_run(fn_before, ())
    if callable(fn_after):
        meas.add_after_run(fn_after, ())

    for p in dependents:
        meas.register_parameter(p, setpoints=("time",))

    with meas.run() as datasaver:
        start = time.time()
        for event in event_list:
            begin = time.time()
            if callable(event.fn_before):
                event.fn_before()
            while (time.time() - begin < event.time):
                save_list = []
                save_list.append(["time", time.time() - start])
                for p in dependents:
                    save_list.append([p, p.get()])

                datasaver.add_result(*save_list)

                time.sleep(event.step)


def linear_trace(independents, dependents, sweep_param, values,
                 delay=None, exp=None, station=None, fn_before=None,
                 fn_after=None):
    """
    Sweep a single variable over a linear range. Allows other params
    to be defined as dependent on this param and measured / saved.
    """

    meas = Measurement(exp=exp, station=station)
    for p in independents:
        meas.register_parameter(p,)
    for p in dependents:
        meas.register_parameter(p, setpoints=(sweep_param,))

    if callable(fn_before):
        meas.add_before_run(fn_before, ())
    if callable(fn_after):
        meas.add_after_run(fn_after, ())

    save_list = []
    for p in (independents + dependents):
        save_list.append([p, None])

    with meas.run() as datasaver:

        for point in values:
            sweep_param.set(point)
            if delay is not None:
                time.sleep(delay)

            for i, p in enumerate(save_list):
                save_list[i][1] = p[0].get()

            datasaver.add_result(*save_list)
