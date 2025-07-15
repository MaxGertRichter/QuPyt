"""
Main measurement loop.
"""

import logging
from time import sleep
from datetime import datetime
from typing import Dict, Any
import gc
import numpy as np
import yaml
from tqdm import tqdm

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.measurement_logic.data_handling import Data
from qupyt.hardware.synchronisers import Synchroniser
from qupyt.hardware.sensors import Sensor
from qupyt._version import __version__ as qupyt_version


def run_measurement(
    static_devices: DeviceHandler,
    dynamic_devices: DynamicDeviceHandler,
    sensor: Sensor,
    synchroniser: Synchroniser,
    params: Dict[str, Any],
) -> str:
    import matplotlib.pyplot as plt
    import msvcrt
    plt.ion()
    fig, (ax, ax_ratio) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    static_devices.set_all_params()
    iterator_size = int(params.get("dynamic_steps", 1))
    mid = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return_status = "all_fail"
    try:
        synchroniser.open()
        synchroniser.stop()
        synchroniser.load_sequence()
        synchroniser.run()
        sleep(0.1)
        sensor.open()
        sleep(0.5)
        while True:
            # timestamp for this run
            mid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{params['experiment_type']}_{mid}"
            print(f"\n=== Starting measurement: {run_name} ===")

            data_container = Data(params["data"])
            data_container.set_dims_from_sensor(sensor)
            data_container.create_array()

            for itervalue in tqdm(range(iterator_size)):
                dynamic_devices.next_dynamic_step()
                sleep(0.1)
                for avg in tqdm(
                    range(int(params["averages"])), leave=itervalue == (iterator_size - 1)
                ):
                    sleep(float(params.get("sleep", 0)))
                    data = sensor.acquire_data(synchroniser)
                    data_container.update_data(data, itervalue, avg)
            params["filename"] = params["experiment_type"] + "_" + mid
            params["measurement_status"] = return_status
            params["qupyt_version"] = qupyt_version
            print(data_container.data.shape)

            # --- Real-time plotting ---
            # data_container.data shape: (2, N, 1, 1) depending whether it is dynamic steps or not  # shape: (2, 1,N,1)+
            data_to_plot = data_container.data
            n_meas = params["sensor"]["config"]["number_measurements"]
            ref = data_to_plot[0].flatten()/(n_meas/2)
            mess = data_to_plot[1].flatten()/(n_meas/2)
            N_dyn = int(params['dynamic_steps'])
            min_f = float(params['dynamic_devices']['mw_source']['config']['frequency'][0]) /1e9
            max_f = float(params['dynamic_devices']['mw_source']['config']['frequency'][1]) /1e9
            step  = (max_f - min_f)/N_dyn
            x_ESR = np.arange(min_f, max_f - step/2, step)
            ax.clear()
            ax.plot(x_ESR, ref, label="Reference")
            ax.plot(x_ESR, mess, label="Measurement")
            ax.set_title(f"Measurement: {run_name}")
            ax.legend()
            fig.canvas.draw()
            plt.pause(0.01)

            # --- Ratio plotting ---
            av = params["averages"]
            light_level = np.average(ref)/av * 1e3
            ratio = mess / ref
            contrast = np.min(ratio)
            ax_ratio.clear()
            ax_ratio.plot(x_ESR, ratio, label="Ratio (Measurement/Reference)")
            ax_ratio.set_title(f"Ratio: {run_name}"+ f", Light level: {light_level:.1f} mV, Contrast: {contrast:.4f}")
            ax_ratio.legend()
            fig.canvas.draw()
            plt.pause(0.01)
            # --- End plotting ---

            # Wait for user to press Enter to continue or 'q' to quit
            cmd = input("Press Enter to run the next measurement, Press 's' to save data and run the next measurement, or 'q' + Enter to quit: ").strip().lower()
            if cmd == "s":
                data_container.save(params["filename"])
                with open(params["filename"] + ".yaml", "w", encoding="utf-8") as file:
                    yaml.dump(params, file)
                del data_container
                gc.collect()
            if cmd == "q":
                print("Exiting measurement loop.")
                break
            synchroniser.stop()
            synchroniser.run()
            dynamic_devices._reset_step_counter()
            sleep(0.1)
    except Exception as e:
        print(f"exc {e}")
        logging.exception("An error occured during the measurement!")
        return_status = "failed"
    finally:
        sensor.close()
        synchroniser.close()
        print("sensor closed")
        return_status = "success"
    return return_status
