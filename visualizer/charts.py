import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1 — CUDA: среднее время vs размер блока
# ============================================================
def plot_cuda_blocksize(cuda_results, video_label, save=False, save_dir="./stats/"):
    block_sizes = sorted(cuda_results.keys())
    avg_times = [np.mean(cuda_results[bs]) for bs in block_sizes]

    plt.figure()
    plt.plot(block_sizes, avg_times, marker="o")
    plt.xlabel("CUDA block size")
    plt.ylabel("Average time (s)")
    plt.title(f"CUDA: avg time vs block size ({video_label})")
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/cuda_blocksize_{video_label}.png")
        plt.close()
    else:
        return plt.gcf()


# ============================================================
# 2 — OpenCL: среднее время vs workgroup size
# ============================================================
def plot_opencl_workgroupsize(opencl_results, video_label, save=False, save_dir="./stats/"):
    wg_sizes = sorted(opencl_results.keys())
    avg_times = [np.mean(opencl_results[wg]) for wg in wg_sizes]

    plt.figure()
    plt.plot(wg_sizes, avg_times, marker="o")
    plt.xlabel("OpenCL workgroup size")
    plt.ylabel("Average time (s)")
    plt.title(f"OpenCL: avg time vs workgroup size ({video_label})")
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/opencl_workgroup_{video_label}.png")
        plt.close()
    else:
        return plt.gcf()


# ============================================================
# 3 — CUDA vs OpenCL: средние времена (НОВЫЙ ГРАФИК)
# ============================================================
def plot_cuda_vs_opencl_avg(cuda_results, opencl_results, video_label, save=False, save_dir="./stats/"):
    avg_cuda = np.mean([np.mean(v) for v in cuda_results.values()])
    avg_opencl = np.mean([np.mean(v) for v in opencl_results.values()])

    labels = ["CUDA avg", "OpenCL avg"]
    values = [avg_cuda, avg_opencl]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Average time (s)")
    plt.title(f"CUDA vs OpenCL — AVERAGE ({video_label})")
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/cuda_vs_opencl_avg_{video_label}.png")
        plt.close()
    else:
        return plt.gcf()


# ============================================================
# 4 — Глобальный график средних значений CUDA vs OpenCL (НОВЫЙ)
# ============================================================
def plot_all_videos_avg(cuda_data_all, opencl_data_all, save=False, save_dir="./stats/"):
    video_labels = ["15s", "30s", "60s"]

    avg_cuda = []
    avg_opencl = []

    for label in video_labels:
        cuda_group = [v for name, v in cuda_data_all.items() if name.endswith(label)]
        opencl_group = [v for name, v in opencl_data_all.items() if name.endswith(label)]

        if cuda_group:
            avg_cuda.append(
                np.mean([np.mean([np.mean(vals) for vals in d.values()]) for d in cuda_group])
            )
        else:
            avg_cuda.append(np.nan)

        if opencl_group:
            avg_opencl.append(
                np.mean([np.mean([np.mean(vals) for vals in d.values()]) for d in opencl_group])
            )
        else:
            avg_opencl.append(np.nan)

    x = np.arange(len(video_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, avg_cuda, width, label="CUDA avg")
    plt.bar(x + width/2, avg_opencl, width, label="OpenCL avg")

    plt.xticks(x, video_labels)
    plt.ylabel("Average time (s)")
    plt.title("CUDA vs OpenCL (AVERAGE) for all video lengths")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/global_cuda_vs_opencl_avg.png")
        plt.close()
    else:
        return plt.gcf()


def plot_all_block_vs_time(results, save=False, save_path="./stats/all_times.png"):
    """
    Строит единый график:
    CUDA 15s, 30s, 60s
    OpenCL 15s, 30s, 60s
    """

    # ---- Подготовка структур ----
    cuda_data = {"15s": {}, "30s": {}, "60s": {}}
    opencl_data = {"15s": {}, "30s": {}, "60s": {}}

    for r in results:
        video = r["video"]
        duration = float(r["duration"])
        suffix = video.split("_")[-1]   # "15s", "30s", "60s"
        method = r["method"]

        # CUDA ----------------------------------
        if "cuda" in method:
            block_dim = r["block_dim"]
            if block_dim:
                block_size = int(block_dim[0] * block_dim[1])
                cuda_data[suffix].setdefault(block_size, [])
                cuda_data[suffix][block_size].append(duration)

        # OpenCL --------------------------------
        elif "opencl" in method:
            wg = int(r["global_threads"])
            opencl_data[suffix].setdefault(wg, [])
            opencl_data[suffix][wg].append(duration)

    # ---- Построение графика ----
    plt.figure(figsize=(10, 6))

    # CUDA линии
    colors_cuda = {"15s": "red", "30s": "orange", "60s": "brown"}
    for suffix in ["15s", "30s", "60s"]:
        if cuda_data[suffix]:
            xs = sorted(cuda_data[suffix].keys())
            ys = [np.mean(cuda_data[suffix][x]) for x in xs]
            plt.plot(xs, ys, marker="o", color=colors_cuda[suffix], label=f"CUDA {suffix}")

    # OpenCL линии
    colors_opencl = {"15s": "blue", "30s": "cyan", "60s": "navy"}
    for suffix in ["15s", "30s", "60s"]:
        if opencl_data[suffix]:
            xs = sorted(opencl_data[suffix].keys())
            ys = [np.mean(opencl_data[suffix][x]) for x in xs]
            plt.plot(xs, ys, marker="s", linestyle="--", color=colors_opencl[suffix], label=f"OpenCL {suffix}")

    plt.xlabel("Block size / Workgroup size")
    plt.ylabel("Average time (s)")
    plt.title("CUDA & OpenCL: Average time vs block/workgroup size for 15s / 30s / 60s")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()