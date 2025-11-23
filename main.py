from video_io import load_video, save_video

def main():
    input_path = "./vids-original/"
    videos = ["video1_15s.mp4"]  # список видео для обработки
    output_path = "./vids-result/"
    n = 4           # число квантов

    frames = load_video(f"{input_path}{videos[0]}")
    print(f"Загрузил видео из {videos[0]}, количество кадров: {len(frames)}")

    # обработка видоса какая-то
    

    save_video(frames, f"{output_path}{videos[0]}")
    print(f"Сохранил обработанное видео в {output_path}{videos[0]}")
    
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