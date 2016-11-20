import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
import cv2
from scipy.misc import *
from device_choose import *

#default camera
camera = 0

# Kernel function
kernel_src = 'mixture-of-gaussian.cl'

#choose INTEL_PLATFORM or NVIDIA_PLATFORM
pref_platform = NVIDIA_PLATFORM
#choose GPU_DEVICE or CPU_DEVICE
pref_device = GPU_DEVICE 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Choose graphic device and create context for it
    ctx = cl.Context([device_choose(pref_platform, pref_device)])
    mf  = cl.mem_flags

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    #Kernel function instantiation
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()

    mixture_data_buff = np.zeros(10000000, dtype=np.float32)
    params_list = [2.5, 0.5, 0.02, 75.0, 25.0]
    mog_params = np.array(params_list, dtype=np.float32)
    alpha = 0.1

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	#Allocate memory for variables on the device
    mixture_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mixture_data_buff)
    mog_params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mog_params)
    
    cap = cv2.VideoCapture(camera)
    
    while(True):
    	#Read in image
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img_g = cl.image_from_array(ctx, img, 4, mode = 'r', norm_int = True)
            img_shape = (img.shape[1], img.shape[0])

            result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img_shape)

            # Call Kernel. Automatically takes care of block/grid distribution
            prg.mog_image(queue, img_shape, None, img_g, result_g, mixture_data_g, mog_params_g, np.float32(alpha))
            
            result = np.empty_like(img)
            cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img_shape)
            
            cv2.imshow('output', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()