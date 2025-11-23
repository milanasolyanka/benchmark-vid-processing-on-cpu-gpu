import numpy as np
from numba import cuda

# CUDA kernel
@cuda.jit
def posterize_kernel(frames, lut):
    """
    Обрабатывает массив кадров RGB:
    frames: uint8 array (num_frames, H, W, 3)
    lut: uint8 array (256)

    Сетка: 2D (height x width)
    Блоки: 2D (blockDim.x = width, blockDim.y = height)
    """

    # 2D индексы ядер GPU
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    num_frames = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]

    if x >= width or y >= height:
        return

    #применяем LUT
    for f in range(num_frames):
        r = frames[f, y, x, 0]
        g = frames[f, y, x, 1]
        b = frames[f, y, x, 2]

        frames[f, y, x, 0] = lut[r]
        frames[f, y, x, 1] = lut[g]
        frames[f, y, x, 2] = lut[b]


# хост-функция
def process_video_cuda_grid2d_block2d(frames: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Обрабатывает видео массив frames (N, 720, 1280, 3) на GPU с использованием:
    - Grid: 2D
    - Blocks: 2D

    Возвращает новый массив кадров.
    """

    #Проверки
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")

    if lut.dtype != np.uint8 or lut.shape[0] != 256:
        raise ValueError("lut must be uint8 with length 256")

    num_frames, height, width, _ = frames.shape

    #кидаем массив на VRAM
    d_frames = cuda.to_device(frames)
    d_lut = cuda.to_device(lut)

    # сетка cuda (2d grid, 2d block)
    block_dim = (16, 16)     
    grid_dim = (
        (width + block_dim[0] - 1) // block_dim[0],
        (height + block_dim[1] - 1) // block_dim[1]
    )

    #выполняем функцию на kernels
    posterize_kernel[grid_dim, block_dim](d_frames, d_lut)
    cuda.synchronize()

    #кидаем результат на RAM
    result = d_frames.copy_to_host()
    return result, {
    "block_dim": block_dim,
    "grid_dim": grid_dim,
    "global_threads": grid_dim[0] * grid_dim[1] * block_dim[0] * block_dim[1]
}
