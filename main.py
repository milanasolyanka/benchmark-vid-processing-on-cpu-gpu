from processing import build_lut
from processing import process_video_cuda_grid2d_block2d, process_video_cuda_grid3d_block2d
from processing import process_video_opencl_1d, process_video_opencl_2d
from video_io import load_video, save_video

def main():
    input_path = "./vids-original/"
    videos = ["video1_15s", "video1_30s", "video1_60s",
              "video2_15s", "video2_30s", "video2_60s",
              "video3_15s", "video3_30s", "video3_60s"]  # список видео для обработки
    output_path = "./vids-result/"
    n = 10           # число квантов

    frames = load_video(f"{input_path}{videos[0]}.mp4")
    print(f"Загрузил видео из {videos[0]}, количество кадров: {len(frames)}")

    # обработка видоса
    lut = build_lut(n)  # создали lookup таблицы
    # result = process_video_cuda_grid3d_block2d(frames, lut)
    result = process_video_opencl_2d(frames, lut)

    save_video(result, f"{output_path}{videos[0]}_n_{n}.mp4")
    print(f"Сохранил обработанное видео в {output_path}{videos[0]}")
    
    # первоначальный алгоритм от чатгпт
    # lut = build_lut(n)  # создали lookup таблицы
    
    # results = []
    # for video_path in videos:
    #     frames = load_video(video_path)
    #     processed = process_cuda_2d(frames, lut)
    #     save_video(processed, video_path + "_cuda2d.mp4")
    #     results.append(...)
    
    # plot_execution_times(results)

if __name__ == "__main__":
    main()