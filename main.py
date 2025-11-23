import time
import json
# import numpy as np
from processing import build_lut
from processing import (
    process_video_cuda_grid2d_block2d,
    process_video_cuda_grid3d_block2d,
    process_video_opencl_1d,
    process_video_opencl_2d
)
from video_io import load_video, save_video
from visualizer import (
    plot_cuda_blocksize,
    plot_opencl_workgroupsize,
    plot_cuda_vs_opencl_best,
    plot_all_videos
)
import os

def generate_all_charts(json_path: str, output_dir: str = "./stats/"):
    """
    Загружает results.json, агрегирует данные и строит PNG-графики.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Загружаем результаты
    with open(json_path, "r") as f:
        results = json.load(f)

    # --- СЛОВАРИ ДЛЯ АГРЕГАЦИИ ---
    # cuda_data["video_name"][block_size] = [times...]
    cuda_data = {}

    # opencl_data["video_name"][workgroup_size] = [times...]
    opencl_data = {}

    # --- РАЗБИРАЕМ СТРОКИ ---
    for r in results:
        video = r["video"]
        method = r["method"]
        duration = r["duration"]

        # CUDA grid2d_block2d / grid3d_block2d
        if "cuda" in method:
            block_dim = tuple(r["block_dim"])   # например (16,16)
            block_size = block_dim[0] * block_dim[1]

            cuda_data.setdefault(video, {})
            cuda_data[video].setdefault(block_size, [])
            cuda_data[video][block_size].append(duration)

        # OpenCL 1D / 2D
        elif "opencl" in method:
            wg = r["global_threads"]  # например 64
            opencl_data.setdefault(video, {})
            opencl_data[video].setdefault(wg, [])
            opencl_data[video][wg].append(duration)

    # ===============================
    #  ПОСТРОЕНИЕ ГРАФИКОВ
    # ===============================
    for video in cuda_data.keys() | opencl_data.keys():
        short_label = video.split("_")[-1]  # "15s", "30s", "60s"

        # 1) CUDA block size
        if video in cuda_data:
            fig = plot_cuda_blocksize(cuda_data[video], short_label, save=True, save_dir=output_dir)

        # 2) OpenCL workgroup size
        if video in opencl_data:
            fig = plot_opencl_workgroupsize(opencl_data[video], short_label, save=True, save_dir=output_dir)

        # 3) CUDA vs OpenCL best
        if video in cuda_data and video in opencl_data:
            fig = plot_cuda_vs_opencl_best(cuda_data[video], opencl_data[video], short_label, save=True, save_dir=output_dir)

    # 4) Глобальный график для трёх длительностей
    plot_all_videos(cuda_data, opencl_data, save=True, save_dir=output_dir)

    print(f"Графики сохранены в {output_dir}")


def main():
    input_path = "./vids-original/"
    output_path = "./vids-result/"

    videos = [
        "video1_15s", "video1_30s", "video1_60s",
        "video2_15s", "video2_30s", "video2_60s",
        "video3_15s", "video3_30s", "video3_60s"
    ]

    n = 10   # число квантов
    lut = build_lut(n)

    # Методы обработки имя : функция
    processing_methods = {
        "cuda_grid2d_block2d": process_video_cuda_grid2d_block2d,
        "cuda_grid3d_block2d": process_video_cuda_grid3d_block2d,
        "opencl_1d": process_video_opencl_1d,
        "opencl_2d": process_video_opencl_2d
    }
    # повторяем каждый метод 3 раза
    num_iterations = 3

    results = []

    # для каждого видео 
    for video_name in videos:
        input_file = f"{input_path}{video_name}.mp4"

        print(f"\nЗагружаю видео {video_name}...")
        frames = load_video(input_file)
        print(f"{video_name}: кадров = {len(frames)}")

        # для каждого метода
        for method_name, method_fn in processing_methods.items():

            print(f"\nЗапускаю обработку: {method_name} для {video_name}")

            durations = []

            # три запуска
            for i in range(1, num_iterations + 1):
                print(f"  Итерация {i} начинается")

                start = time.perf_counter()
                
                # обработка
                processed, meta = method_fn(frames, lut)

                end = time.perf_counter()

                # сохранение результата
                output_file = (
                    f"{output_path}"
                    f"{video_name}__{method_name}__iter{i}__n_{n}.mp4"
                )

                save_video(processed, output_file)

                durations.append(end - start)

                results.append({
                    "video": video_name,
                    "method": method_name,
                    "iteration": i,
                    "duration": end - start,
                    "duration_avg": None,  
                    "block_dim": meta["block_dim"],
                    "grid_dim": meta["grid_dim"],
                    "global_threads": meta["global_threads"],
                })

                print(f"Сохранен файд: {output_file}")
            
            avg = sum(durations) / len(durations)

            # записываем среднее
            for r in results[-3:]:
                r["duration_avg"] = avg

    print("\nВСЕ ОБРАБОТАНО УРА")
    # сохраняем как json для анализа matplotlib
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Результаты сохранены в results.json")
    generate_all_charts("results.json")
    print("ВСЕ ГРАФИКИ ПОСТРОЕНЫ УРА")


if __name__ == "__main__":
    main()
