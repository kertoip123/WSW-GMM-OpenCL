import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
import cv2
import time
from scipy.misc import *
from device_choose import *

#default camera
camera = 0
resolution = 640*480

nmixtures = 5
alpha = 0.1
k = 2.5
T = 0.4
init_var = 15.0
min_var = 0.0

#choose INTEL_PLATFORM or NVIDIA_PLATFORM
pref_platform = NVIDIA_PLATFORM
#choose GPU_DEVICE or CPU_DEVICE
pref_device = GPU_DEVICE 

# Kernel function
kernel_src = 'mixture-of-gaussian.cl'


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

    mixture_data_buff = np.zeros(3*nmixtures*resolution, dtype=np.float32)
    mixture_data_buff[0:resolution*nmixtures] = 1.0/nmixtures/10
    mixture_data_buff[resolution*nmixtures+1:2*resolution*nmixtures] = init_var
    
    params_list = [k, T, init_var, min_var]
    mog_params = np.array(params_list, dtype=np.float32)
    

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	#Allocate memory for variables on the device
    mixture_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mixture_data_buff)
    mog_params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mog_params)
    
    cap = cv2.VideoCapture(camera)
    
    time_begin = time.time()
    cnt = 0
    
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
            
            time_current = time.time()
            if time_current-time_begin >= 5:
                avg_fps = cnt/(time_current - time_begin)
                time_begin = time_current
                cnt = 0
                logging.info('Average fps: %.1f' % avg_fps)
            else:
                cnt += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()