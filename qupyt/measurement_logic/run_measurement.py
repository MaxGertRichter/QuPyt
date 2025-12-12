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
from evaluation_package import casr as casr
from evaluation_package import utils as ut


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
    fig, ((ax, ax_ratio), (ax_ref, ax_mess)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
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

            # --- Real-time plotting ---
            # data_container.data shape: (2, N, 1, 1) depending whether it is dynamic steps or not
            data = data_container.data
            contrast = data[1].flatten() - data[0].flatten()
            #contrast = ut.contrast(data)
            rabi_steps = params["sensor"]["config"]["number_measurements"]
            av = params["averages"]

            ref = data[0].flatten()/av

            mess = data[1].flatten()/av
            mask_index = 20
            frequencies = casr.calc_fourier_frequencies(params)[mask_index:]
            fft_spectrum = casr.calc_fourier_transform(params,data)[mask_index:]
            prominence = np.median(fft_spectrum) * 5
            mask, peak_info = casr.noise_only_mask(frequencies, fft_spectrum, prominence=prominence, rel_pad=10, width_hz=1)
            idx, freq, amp = casr.find_peak_near(frequencies, fft_spectrum, 500)
            sensitivity, snr, std = casr.calc_sensitivity(params, data, window_hz=50, return_snr=True, prominence=prominence)

            ax.clear()
            ax.plot(ref, label="Reference")
            ax.plot(mess, label="Measurement")
            ax.set_title("Measurements")
            ax.set_xlabel("t (ns)")
            ax.legend()
            # --- Ratio plotting ---
            ax_ratio.clear()
            ax_ratio.plot(frequencies,fft_spectrum, label="CASR Spectrum")
            ax_ratio.plot(
                freq, amp, "ro",
                label=(
                    f"Calibration signal at {freq:.1f} Hz\n"
                    f"with 10nT amplitude,\n"
                    f"SNR={snr:.1f},\n"
                    f"std={std:.3e},\n"
                    f"sensitivity={sensitivity*1e12:.1f} pT/√Hz, \n"
                )
            )
            ax_ratio.plot(frequencies[mask], fft_spectrum[mask], label="noise")
            ax_ratio.set_title("CASR Spectrum")
            ax_ratio.set_xlabel("Frequency [Hz]")
            ax_ratio.set_ylabel("FFT amplitude [a.u.]")
            ax_ratio.legend()
            
            # --- Reference spectrum plotting ---
            data0_ref = data[0].flatten()
            fft_spectrum0_ref = casr.calc_fourier_transform(params, data0_ref, contrast=False)[mask_index:]
            prominence_ref = np.median(fft_spectrum0_ref) * 5
            mask_ref, peak_info_ref = casr.noise_only_mask(frequencies, fft_spectrum0_ref, prominence=prominence_ref, rel_pad=10, width_hz=1)
            idx_ref, freq_ref, amp_ref = casr.find_peak_near(frequencies, fft_spectrum0_ref, 500)
            sensitivity_ref, snr_ref, std_ref = casr.calc_sensitivity(params, data0_ref, window_hz=50, return_snr=True, prominence=prominence_ref, contrast=False)
            sensitivity_ref = sensitivity_ref * 1/np.sqrt(2)
            
            ax_ref.clear()
            ax_ref.plot(frequencies, fft_spectrum0_ref)
            ax_ref.plot(
                freq_ref, amp_ref, "ro",
                label=(
                    f"Calibration signal at {freq_ref:.1f} Hz\n"
                    f"with 10nT amplitude,\n"
                    f"SNR={snr_ref:.1f},\n"
                    f"std={std_ref:.3e},\n"
                    f"sensitivity={sensitivity_ref*1e12:.1f} pT/√Hz"
                )
            )
            ax_ref.plot(frequencies[mask_ref], fft_spectrum0_ref[mask_ref], label="noise")
            ax_ref.set_title("Reference Spectrum")
            ax_ref.set_xlabel("Frequency [Hz]")
            ax_ref.set_ylabel("FFT amplitude [a.u.]")
            ax_ref.legend()
            
            # --- Measurement spectrum plotting ---
            data0_mess = data[1].flatten()
            fft_spectrum0_mess = casr.calc_fourier_transform(params, data0_mess, contrast=False)[mask_index:]
            prominence_mess = np.median(fft_spectrum0_mess) * 5
            mask_mess, peak_info_mess = casr.noise_only_mask(frequencies, fft_spectrum0_mess, prominence=prominence_mess, rel_pad=10, width_hz=1)
            idx_mess, freq_mess, amp_mess = casr.find_peak_near(frequencies, fft_spectrum0_mess, 500)
            sensitivity_mess, snr_mess, std_mess = casr.calc_sensitivity(params, data0_mess, window_hz=50, return_snr=True, prominence=prominence_mess, contrast=False)
            sensitivity_mess = sensitivity_mess * 1/np.sqrt(2)
            
            ax_mess.clear()
            ax_mess.plot(frequencies, fft_spectrum0_mess)
            ax_mess.plot(
                freq_mess, amp_mess, "ro",
                label=(
                    f"Calibration signal at {freq_mess:.1f} Hz\n"
                    f"with 10nT amplitude,\n"
                    f"SNR={snr_mess:.1f},\n"
                    f"std={std_mess:.3e},\n"
                    f"sensitivity={sensitivity_mess*1e12:.1f} pT/√Hz"
                )
            )
            ax_mess.plot(frequencies[mask_mess], fft_spectrum0_mess[mask_mess], label="noise")
            ax_mess.set_title("Measurement Spectrum")
            ax_mess.set_xlabel("Frequency [Hz]")
            ax_mess.set_ylabel("FFT amplitude [a.u.]")
            ax_mess.legend()
            
            # --- Figure title ---
            fig.suptitle(f"File: {params['filename']}")
            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(0.01)
            # --- End plotting ---

            # don't save data for now when we run in continous mode
            #data_container.save(params["filename"])
            #with open(params["filename"] + ".yaml", "w", encoding="utf-8") as file:
            #    yaml.dump(params, file)
            

            # Check for keypress to exit
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key.lower() == b'q':
                    data_container.save(params["filename"])
                    with open(params["filename"] + ".yaml", "w", encoding="utf-8") as file:
                        yaml.dump(params, file)
                    del data_container
                    gc.collect()
                    print("Exiting measurement loop and saving data.")
                    break
            del data_container
            gc.collect()
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
