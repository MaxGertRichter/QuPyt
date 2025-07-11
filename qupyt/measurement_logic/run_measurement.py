"""
Main measurement loop.
"""

import logging
from time import sleep
from datetime import datetime
from typing import Dict, Any
import gc

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
    fig, ax = plt.subplots()
    fig_ratio, ax_ratio = plt.subplots()
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
            # data_container.data shape: (2, N, 1, 1)
            data_to_plot = data_container.data[:, :, 0, 0]  # shape: (2, N)
            x = range(data_to_plot.shape[1])
            ax.clear()
            ax.plot(x, data_to_plot[0], label="Series 1")
            ax.plot(x, data_to_plot[1], label="Series 2")
            ax.set_title(f"Measurement: {run_name}")
            ax.legend()
            plt.draw()
            plt.pause(0.01)

            # --- Ratio plotting ---
            ratio = data_to_plot[1] / data_to_plot[0]
            ax_ratio.clear()
            ax_ratio.plot(x, ratio, label="Ratio (Series 2 / Series 1)")
            ax_ratio.set_title(f"Ratio: {run_name}")
            ax_ratio.legend()
            fig_ratio.canvas.draw()
            plt.pause(0.01)
            # --- End plotting ---

            #data_container.save(params["filename"])
            #with open(params["filename"] + ".yaml", "w", encoding="utf-8") as file:
            #    yaml.dump(params, file)
            del data_container
            gc.collect()

            # Check for keypress to exit
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key.lower() == b'q':
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
