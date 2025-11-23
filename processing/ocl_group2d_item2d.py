import numpy as np
import pyopencl as cl

# 2D OpenCL Kernel
KERNEL_2D = r"""
__kernel void posterize_2d(
    __global uchar4 *frames,
    __global uchar *lut,
    const int width,
    const int height,
    const int num_frames
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    for (int f = 0; f < num_frames; f++) {

        int idx = (f * width * height) + (y * width + x);

        uchar4 px = frames[idx];

        px.x = lut[px.x];
        px.y = lut[px.y];
        px.z = lut[px.z];

        frames[idx] = px;
    }
}
"""

# хост
def process_video_opencl_2d(frames: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    OpenCL NDRange 2D:
        - global:  (width, height)
        - local:   (workgroup_x, workgroup_y)

    frames: (num_frames, H, W, 3), uint8
    """

    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")

    num_frames, H, W, _ = frames.shape
    total_pixels = num_frames * H * W

    flat = np.zeros(total_pixels, dtype=np.uint32).view(np.uint8).reshape(-1, 4)
    flat[:, :3] = frames.reshape(-1, 3)
    flat = flat.view(np.uint32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, KERNEL_2D).build()

    mf = cl.mem_flags
    d_frames = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flat)
    d_lut = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut)

    global_size = (W, H)
    local_size = None 

    program.posterize_2d(queue, global_size, local_size,
                         d_frames, d_lut,
                         np.int32(W), np.int32(H), np.int32(num_frames))

    cl.enqueue_copy(queue, flat, d_frames).wait()

    out = flat.view(np.uint8).reshape(-1, 4)[:, :3]
    out = out.reshape(num_frames, H, W, 3)

    return out
