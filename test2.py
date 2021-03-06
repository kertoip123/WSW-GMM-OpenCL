import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
import cv2
from scipy.misc import *
from device_choose import *

#Test sequence constants
frame_num = 1700
rel_path  = './input/in00%04d.jpg'
dest_path = './output/out00%04d.jpg' 

nmixtures = 10

# Kernel function
kernel_src = 'test2.cl'

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Choose graphic device and create context for it
    ctx = cl.Context([device_choose(NVIDIA_PLATFORM, GPU_DEVICE)])
    mf  = cl.mem_flags

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    #Kernel function instantiation
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()

    params_list = [1, 0, 0]
    params = np.array(params_list, dtype=np.float32)
    params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=params)

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)

    cap = cv2.VideoCapture(0)
    
    while(True):
    	#Read in image
    	#img = imread(rel_path % i, flatten=False, mode = 'RGBA')
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img_shape = (img.shape[1], img.shape[0])

            img_g = cl.image_from_array(ctx, img, 4, mode = 'r', norm_int = True)
            result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img_shape)

            # Call Kernel. Automatically takes care of block/grid distribution
            prg.test(queue, img_shape, None, img_g, result_g, params_g)
            
            result = np.empty_like(img)
            cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img_shape)
            
            cv2.imshow('output', result)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()