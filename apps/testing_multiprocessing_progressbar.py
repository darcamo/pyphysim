import multiprocessing

import numpy as np

from pyphysim.progressbar import ProgressbarMultiProcessServer


def func(rep_max, progressbar):
    for i in range(rep_max):
        a = np.random.randn(3, 3)
        b = np.random.randn(3, 3)
        c = np.linalg.inv(a @ b)
        progressbar.progress(i)
    return c


if __name__ == '__main__':

    pb = ProgressbarMultiProcessServer(message="Running")

    num_process = 4

    rep_max = 100000

    procs = []
    for i in range(num_process):
        proc_args = [
            rep_max,
            pb.register_client_and_get_proxy_progressbar(rep_max)
        ]
        procs.append(multiprocessing.Process(target=func, args=proc_args))

    # xxxxx Start all processes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for proc in procs:
        proc.start()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Start the processbar xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pb.start_updater()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Join all processes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for proc in procs:
        proc.join()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Stop the processbar xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pb.stop_updater()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
