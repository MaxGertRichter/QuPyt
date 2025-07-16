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
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
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
            #data_container._set_reference_channels(1)
            data_container.set_dims_from_sensor(sensor)
            data_container.create_array()
            print("Shape at creation of data container: "+str(data_container.data.shape))
        
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
    
            plot_num = data_container.data.shape[0]
            data_to_plot = data_container.data
            print("Shape of data to plot: "+str(data_to_plot.shape))
            # --- Plotting ---
            
            ax.clear()
            for i in range(plot_num):
                ax.plot(data_to_plot[i].flatten(), label=f"Plot {i}")
            ax.legend()
            ax.set_title(f"Measurement: {run_name}")
            fig.canvas.draw()
            plt.pause(0.01)

            cmd = input("\nPress Enter to repeat measurement, 's' to save data and run the next measurement, or 'q' + Enter to quit: ").strip().lower()
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
