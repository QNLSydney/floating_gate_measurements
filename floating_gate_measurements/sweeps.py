import time
import numpy as np
from scipy import polyfit
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.data_set import load_by_id
import matplotlib.pyplot as plt


class event():
    def __init__(self, time, step, fn_before):
        self.time = time
        self.step = step
        self.fn_before = fn_before


def awg_setup(awg):
    """
    Setup the 'dc' signal to control the relay
    """
    awg.ch1.frequency(1)
    awg.ch1.function_type('SIN')
    awg.ch1.frequency_mode('CW')
    awg.ch1.amplitude_unit('VPP')
    awg.ch1.amplitude('10e-3')
    awg.ch1.offset('4.95')


def setup(station, vals):
    station.b2962.VDD(vals["vdd"])
    station.b2962.VDRAIN(vals["vdrain"])
    station.n6705b.VBUS(vals["vbus"])
    station.n6705b.VFEEDBACK(vals["vfeedback"])
    station.n6705b.VREF(vals["vref"])
    station.n6705b.VTUN(vals["vtun"])
    station.yoko.VBIAS(vals["vbias"])
    station.yoko.source_mode("VOLT")
    station.yoko.auto_range(False)
    station.yoko.voltage_range(10)
    station.b2962.ch1.current_limit(10e-3)
    station.b2962.ch2.current_limit(10e-3)
    awg_setup(station.awg)
    all_enable(station)

def all_enable(station):
    for ch in station.n6705b.channels:
        ch.enable("on")
    station.b2962.ch1.enable('on')
    station.b2962.ch2.enable('on')
    station.yoko.output('on')

def all_disable(station):
    for ch in station.n6705b.channels:
        ch.enable("off")
    station.b2962.ch1.enable('off')
    station.b2962.ch2.enable('off')
    station.yoko.output('off')

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
            time.sleep(0.5)
            while (time.time() - begin < event.time):
                save_list = []
                save_list.append(["time", time.time() - start])
                for p in dependents:
                    save_list.append([p, p.get()])

                datasaver.add_result(*save_list)

                time.sleep(event.step)
        runid = datasaver.run_id
    return runid


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

        runid = datasaver.run_id
    return runid

def find_trip_voltage(x,y):
    """
    if we assume that the trip point is the median Y voltage, we can
    just return the corresponding X value
    """
    m = max(y) - ((max(y)-min(y))/2)

    s = sorted(list(zip(x,y)),
               key = lambda x: np.abs(x[1] - m))

    return s[0][0]

def find_vref(x,y):
    mid_point = max(y) - ((max(y)-min(y))/2)
    s = sorted(list(zip(x,y)),
               key = lambda x: np.abs(x[1] - mid_point))

    return s[0][0]

def comparator_sweep(exp, station, fn_switch, voltages, values=None):
    """
    open loop sweep of vref
    """
    # Set the switch state to 'open'
    if not callable(fn_switch):
        raise ValueError("Expecting Switch Function")
    fn_switch("open")

    deps = [station.dmm.VOUT]
    indeps = [station.n6705b.VBUS,
              station.n6705b.VFEEDBACK,
              station.n6705b.VREF,
              station.n6705b.VTUN,
              station.b2962.VDD,
              station.b2962.VDRAIN,
              station.yoko.VBIAS,
              station.b2962.ch1.current,
              station.b2962.ch2.current]

    if values is None:
        values = np.linspace(0, voltages['vdd'], 101)
    # Sweep VFEEDBACK
    # runid = linear_trace(indeps, deps, station.n6705b.VFEEDBACK,
    #                       np.linspace(0, voltages['vdd'], 101), delay=0.5,
    #                       exp=exp, station=station,
    #                       fn_before=lambda: setup(station, voltages),
    #                       fn_after=lambda: all_disable(station))
    # # determine trip voltage
    # # pull "dmm_volt" and "n6705b_VFEEDBACK" out of dataset
    # data = load_by_id(runid)
    # sweep_data = data.get_parameter_data()
    # trip_point = find_trip_voltage(sweep_data['dmm_VOUT']['n6705b_VFEEDBACK'],
    #                                sweep_data['dmm_VOUT']['dmm_VOUT'])

    # Determine a good vref
   # voltages['vfeedback'] = trip_point
    runid = linear_trace(indeps, deps, station.n6705b.VREF,
                          values, delay=0.5,
                          exp=exp, station=station,
                          fn_before=lambda: setup(station, voltages),
                          fn_after=lambda: all_disable(station))
    data = load_by_id(runid)
    sweep_data = data.get_parameter_data()
    vref_point = find_vref(sweep_data['dmm_VOUT']['n6705b_VREF'],
                           sweep_data['dmm_VOUT']['dmm_VOUT'])
    return (vref_point, runid)


def measure_tunneling_rate(exp, station, fn_switch, voltages, vtun_vals, time_step=60):
    """
    open loop sweep of vref
    """
    # Set the switch state to 'open'
    if not callable(fn_switch):
        raise ValueError("Expecting Switch Function")
    fn_switch("close")


    deps = [station.dmm.VOUT,
            station.n6705b.VBUS,
            station.n6705b.VFEEDBACK,
            station.n6705b.VREF,
            station.n6705b.VTUN,
            station.b2962.VDD,
            station.b2962.VDRAIN,
            station.yoko.VBIAS]
    # Sweep VTUN and measure tunneling rate
    runs = []
    for vtun in vtun_vals:
        events = [event(time_step, 0.5, lambda: station.n6705b.VTUN(vtun))]
        runid = time_sweep(deps, events,
                           fn_before=lambda: setup(station, voltages),
                           fn_after=lambda: all_disable(station),
                           exp=exp, station=station)
        runs.append(runid)
    return runs

def _set_inject_volts(station, pmos, gate, limit):
    station.n6705b.VBUS(pmos)
    station.n6705b.VFEEDBACK(gate)
    station.b2962.ch1.current_limit(limit)

def calculate_tunneling_rate(runids, station=None, exp=None, plot=False):
    meas = Measurement(exp=exp, station=station)
    meas.register_parameter(station.n6705b.VTUN,)
    meas.register_custom_parameter("Tunneling_Gradient",
                                   setpoints=(station.n6705b.VTUN,),
                                   unit="V/S")
    meas.register_custom_parameter("runid")

    grads = []
    tun_vs = []
    for id in runids:
        data = load_by_id(id)
        sweep_data = data.get_parameter_data()
        x = sweep_data['dmm_VOUT']['time']
        y = sweep_data['dmm_VOUT']['dmm_VOUT']
        tun_v = sweep_data['n6705b_VTUN']['n6705b_VTUN'][0]
        # y = mx + b
        m,b = polyfit(x,y,1)
        grads.append(m)
        tun_vs.append(tun_v)
        if plot:
            plt.figure()
            plt.plot(x,y)
            y2 = x*m + b
            plt.plot(x,y2)
            plt.ylabel("vout [V]")
            plt.xlabel("time [s]")
            plt.show()

    with meas.run() as datasaver:
            for m, tv, run in zip(grads, tun_vs, runids):
                datasaver.add_result((station.n6705b.VTUN, tv),
                                    ("Tunneling_Gradient", m),
                                    ("runid", run))

def injection(exp, station, fn_switch, events):
    """
    open loop sweep of vref
    """
    voltages = {
    "vdd": 0,
    "vfeedback": 0,
    "vdrain": 0,
    "vbias": 0,
    "vref": 0,
    "vtun": 2,
    "vbus": 2
    }
    # Set the switch state to 'close'
    if not callable(fn_switch):
        raise ValueError("Expecting Switch Function")
    fn_switch("open")


    deps = [station.b2962.IDRAIN]
    indeps = [station.n6705b.VBUS,
                station.n6705b.VFEEDBACK,
                station.n6705b.VREF,
                station.n6705b.VTUN,
                station.b2962.VDD,
                station.b2962.VDRAIN,
                station.yoko.VBIAS]
    station.b2962.ch1.current_limit(1e-3)

    runid = time_sweep(deps, events,
                        fn_before=lambda: setup(station, voltages),
                        fn_after=lambda: all_disable(station),
                        exp=exp, station=station)
    return runid

def tunnel_to_target(exp, station, fn_switch, voltages, vtun , target):
    """
    open loop sweep of vref
    """
    # Set the switch state to 'open'
    if not callable(fn_switch):
        raise ValueError("Expecting Switch Function")
    fn_switch("close")

    deps = [station.dmm.VOUT,
            station.n6705b.VBUS,
            station.n6705b.VFEEDBACK,
            station.n6705b.VREF,
            station.n6705b.VTUN,
            station.b2962.VDD,
            station.b2962.VDRAIN,
            station.yoko.VBIAS]

    meas = Measurement(exp=exp, station=station)
    meas.register_custom_parameter("time", label="Time", unit="S")
    meas.register_custom_parameter("tunnel_target", label="Tunnel Target", unit="V")

    meas.add_before_run(lambda: setup(station, voltages), ())
    meas.add_after_run(lambda: all_disable(station), ())


    for p in deps:
        meas.register_parameter(p, setpoints=("time",))

    voltages['vtun'] = vtun
    with meas.run() as datasaver:
        time.sleep(1)
        start = time.time()
        print(station.dmm.VOUT())
        while (station.dmm.VOUT() > target):
                save_list = []
                save_list.append(["time", time.time() - start])
                save_list.append(["tunnel_target", target])
                for p in deps:
                    save_list.append([p, p.get()])

                datasaver.add_result(*save_list)

                time.sleep(0.1)
        runid = datasaver.run_id
    return runid


def inject_to_target(exp, station, fn_switch, events, target):
    """
    open loop sweep of vref
    """
    # Set the switch state to 'open'
    if not callable(fn_switch):
        raise ValueError("Expecting Switch Function")
    fn_switch("open")
    voltages = {
    "vdd": 0,
    "vfeedback": 0,
    "vdrain": 0,
    "vbias": 0,
    "vref": 0,
    "vtun": 2,
    "vbus": 2
    }
    deps = [station.b2962.IDRAIN,
            station.dmm.VOUT,
            station.n6705b.VBUS,
            station.n6705b.VFEEDBACK,
            station.n6705b.VREF,
            station.n6705b.VTUN,
            station.b2962.VDD,
            station.b2962.VDRAIN,
            station.yoko.VBIAS]

    meas = Measurement(exp=exp, station=station)
    meas.register_custom_parameter("time", label="Time", unit="S")
    meas.register_custom_parameter("injection_target", label="Injection Target", unit="A")

    meas.add_before_run(lambda: setup(station, voltages), ())
    meas.add_after_run(lambda: all_disable(station), ())


    for p in deps:
        meas.register_parameter(p, setpoints=("time",))

    with meas.run() as datasaver:
        time.sleep(1)
        start = time.time()
        events.fn_before()
        while (station.b2962.IDRAIN() > target):
                save_list = []
                save_list.append(["time", time.time() - start])
                save_list.append(["injection_target", target])
                for p in deps:
                    save_list.append([p, p.get()])

                datasaver.add_result(*save_list)

                time.sleep(events.step)
        runid = datasaver.run_id
    return runid
