import numpy as np
from numba import cuda

# CUDA ЯДРО (3D GRID, 2D BLOCKS)
@cuda.jit
def posterize_kernel_3d(frames, lut):
    """
    frames: uint8 array (num_frames, H, W, 3)
    lut: uint8 array (256)

    Grid: 3D (frame_index, y, x)
    Blocks: 2D (threads_y, threads_x)
    """

    # индекс threada
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    f = cuda.blockIdx.z  # один блок один кадр

    num_frames = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]

    # проверка выхода за границы массива
    if f >= num_frames or y >= height or x >= width:
        return

    # постеризация 
    r = frames[f, y, x, 0]
    g = frames[f, y, x, 1]
    b = frames[f, y, x, 2]

    frames[f, y, x, 0] = lut[r]
    frames[f, y, x, 1] = lut[g]
    frames[f, y, x, 2] = lut[b]


# хост-функция
def process_video_cuda_grid3d_block2d(frames: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Обрабатывает кадры с помощью CUDA:

    - Grid: 3D (num_frames, height, width)
    - Blocks: 2D (threads_y, threads_x)

    Возвращает обработанный массив кадров.
    """

    #проверка входных данных
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")

    if lut.dtype != np.uint8 or lut.shape[0] != 256:
        raise ValueError("lut must be uint8 with length 256")

    num_frames, height, width, _ = frames.shape

    #кидаем данные на GPU
    d_frames = cuda.to_device(frames)
    d_lut = cuda.to_device(lut)

    # Настройка grid and block
    block_dim = (16, 16)  # 2D блок
    grid_dim = (
        (width + block_dim[0] - 1) // block_dim[0],   # пиксель по X
        (height + block_dim[1] - 1) // block_dim[1],  #пиксель по Y
        num_frames                                     #один фрейм по Z
    )

    # выполняем функцию на kernels
    posterize_kernel_3d[grid_dim, block_dim](d_frames, d_lut)
    cuda.synchronize()

    #кидаем результат на RAM
    return d_frames.copy_to_host()
