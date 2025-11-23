import numpy as np
import pyopencl as cl

KERNEL_2D = r"""
__kernel void posterize_2d(
    __global uchar4 *frames,
    __global uchar *lut,
    const long width,
    const long height,
    const long num_frames
) {
    long x = get_global_id(0);
    long y = get_global_id(1);

    if (x >= width || y >= height) return;

    for (long f = 0; f < num_frames; f++) {
        long idx = (f * width * height) + (y * width + x);
        uchar4 px = frames[idx];
        px.x = lut[px.x];
        px.y = lut[px.y];
        px.z = lut[px.z];
        frames[idx] = px;
    }
}
"""

def process_video_opencl_2d(frames: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, dict]:
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")

    num_frames, H, W, _ = frames.shape
    total_pixels = np.int64(num_frames * H * W)

    flat = np.zeros(total_pixels, dtype=np.uint32).view(np.uint8).reshape(-1, 4)
    flat[:, :3] = frames.reshape(-1, 3)
    flat = flat.view(np.uint32)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    program = cl.Program(ctx, KERNEL_2D).build()

    mf = cl.mem_flags
    d_frames = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flat)
    d_lut = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut)

    global_size = (np.int64(W), np.int64(H))
    local_size = None

    program.posterize_2d(queue, global_size, local_size,
                         d_frames, d_lut,
                         np.int64(W), np.int64(H), np.int64(num_frames))

    cl.enqueue_copy(queue, flat, d_frames).wait()

    out = flat.view(np.uint8).reshape(-1, 4)[:, :3]
    out = out.reshape(num_frames, H, W, 3)

    return out, {
        "block_dim": local_size,
        "grid_dim": global_size,
        "global_threads": total_pixels
    }