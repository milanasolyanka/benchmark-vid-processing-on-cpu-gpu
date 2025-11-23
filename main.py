import time
import json
from processing import build_lut
from processing import (
    process_video_cuda_grid2d_block2d,
    process_video_cuda_grid3d_block2d,
    process_video_opencl_1d,
    process_video_opencl_2d
)
from video_io import load_video, save_video
import os


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


if __name__ == "__main__":
    main()
