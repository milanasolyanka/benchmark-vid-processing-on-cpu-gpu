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
    plt.title(f"CUDA: vs block size ({video_label})")
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
    plt.title(f"OpenCL: vs workgroup size ({video_label})")
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/opencl_workgroup_{video_label}.png")
        plt.close()
    else:
        return plt.gcf()


# ============================================================
# 3 — CUDA vs OpenCL best
# ============================================================
def plot_cuda_vs_opencl_best(cuda_results, opencl_results, video_label, save=False, save_dir="./stats/"):
    best_cuda = min(np.mean(v) for v in cuda_results.values())
    best_opencl = min(np.mean(v) for v in opencl_results.values())

    labels = ["CUDA best", "OpenCL best"]
    values = [best_cuda, best_opencl]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Average time (s)")
    plt.title(f"CUDA vs OpenCL (best) — {video_label}")
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/cuda_vs_opencl_{video_label}.png")
        plt.close()
    else:
        return plt.gcf()


# ============================================================
# 4 — Глобальный график для 15s / 30s / 60s
# ============================================================
def plot_all_videos(cuda_data_all, opencl_data_all, save=False, save_dir="./stats/"):
    video_labels = ["15s", "30s", "60s"]

    best_cuda = []
    best_opencl = []

    for label in video_labels:
        # ищем видео1_15s, видео2_15s тоже могут быть — берём среднее!
        cuda_group = [v for name, v in cuda_data_all.items() if name.endswith(label)]
        opencl_group = [v for name, v in opencl_data_all.items() if name.endswith(label)]

        # среднее по всем видео одинаковой длины
        if cuda_group:
            best_cuda.append(np.mean([min(np.mean(vals) for vals in d.values()) for d in cuda_group]))
        else:
            best_cuda.append(np.nan)

        if opencl_group:
            best_opencl.append(np.mean([min(np.mean(vals) for vals in d.values()) for d in opencl_group]))
        else:
            best_opencl.append(np.nan)

    x = np.arange(len(video_labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, best_cuda, width, label="CUDA best")
    plt.bar(x + width/2, best_opencl, width, label="OpenCL best")

    plt.xticks(x, video_labels)
    plt.ylabel("Average time (s)")
    plt.title("Best CUDA vs OpenCL for 15s / 30s / 60s")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/global_cuda_vs_opencl.png")
        plt.close()
    else:
        return plt.gcf()
