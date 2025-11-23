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
from visualizer import (
    plot_cuda_blocksize,
    plot_opencl_workgroupsize,
    plot_cuda_vs_opencl_avg,
    plot_all_videos_avg,
    plot_all_block_vs_time
)
import os

def generate_all_charts(json_path: str, output_dir: str = "./stats/"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        results = json.load(f)

    cuda_data = {}
    opencl_data = {}

    for r in results:
        video = r["video"]
        method = r["method"]
        duration = float(r["duration"])

        if "cuda" in method:
            block_dim = r["block_dim"]
            block_size = int(block_dim[0] * block_dim[1]) if block_dim else None
            cuda_data.setdefault(video, {})
            if block_size:
                cuda_data[video].setdefault(block_size, [])
                cuda_data[video][block_size].append(duration)

        elif "opencl" in method:
            wg = int(r["global_threads"])
            opencl_data.setdefault(video, {})
            opencl_data[video].setdefault(wg, [])
            opencl_data[video][wg].append(duration)

    for video in cuda_data.keys() | opencl_data.keys():
        short_label = video.split("_")[-1]

        if video in cuda_data:
            plot_cuda_blocksize(cuda_data[video], short_label, save=True, save_dir=output_dir)

        if video in opencl_data:
            plot_opencl_workgroupsize(opencl_data[video], short_label, save=True, save_dir=output_dir)

        if video in cuda_data and video in opencl_data:
            plot_cuda_vs_opencl_avg(cuda_data[video], opencl_data[video], short_label, save=True, save_dir=output_dir)

    plot_all_videos_avg(cuda_data, opencl_data, save=True, save_dir=output_dir)

    plot_all_block_vs_time(results, save=True)

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

    processing_methods = {
        "cuda_grid2d_block2d": process_video_cuda_grid2d_block2d,
        "cuda_grid3d_block2d": process_video_cuda_grid3d_block2d,
        "opencl_1d": process_video_opencl_1d,
        "opencl_2d": process_video_opencl_2d
    }
    num_iterations = 3
    results = []

    # video_name_flag = 0
    for video_name in videos:
        # video_name = "video3_60s"
        input_file = f"{input_path}{video_name}.mp4"

        print(f"\nЗагружаю видео {video_name}...")
        frames = load_video(input_file)
        print(f"{video_name}: кадров = {len(frames)}")

        for method_name, method_fn in processing_methods.items():
            # method_name = "opencl_2d"
            print(f"\nЗапускаю обработку: {method_name} для {video_name}")
            durations = []

            for i in range(1, num_iterations + 1):
                print(f"  Итерация {i} начинается")
                start = time.perf_counter()
                processed, meta = method_fn(frames, lut)
                end = time.perf_counter()

                output_file = f"{output_path}{video_name}__{method_name}__iter{i}__n_{n}.mp4"
                save_video(processed, output_file)
                durations.append(end - start)

                results.append({
                    "video": video_name,
                    "method": method_name,
                    "iteration": i,
                    "duration": float(end - start),
                    "duration_avg": None,
                    "block_dim": tuple(int(x) for x in meta["block_dim"]) if meta["block_dim"] else None,
                    "grid_dim": tuple(int(x) for x in meta["grid_dim"]) if meta["grid_dim"] else None,
                    "global_threads": int(meta["global_threads"]) if meta["global_threads"] else None,
                })

                print(f"Сохранен файл: {output_file}")

            avg = sum(durations) / len(durations)
            for r in results[-num_iterations:]:
                r["duration_avg"] = float(avg)
            # if method_name == "opencl_2d":
                # break
            
        # if video_name == "video3_60s":
            # break

    print("\nВСЕ ОБРАБОТАНО УРА")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Результаты сохранены в results.json")
    generate_all_charts("results.json")
    print("ВСЕ ГРАФИКИ ПОСТРОЕНЫ УРА")


if __name__ == "__main__":
    main()
