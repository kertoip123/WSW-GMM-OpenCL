import pyopencl as cl
import numpy as np
import logging
from scipy.misc import *

# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

#Test sequence constants
frame_num = 1700
rel_path  = './input/in00%04d.jpg'
dest_path = './output/out00%04d.jpg' 

# Kernel function
kernel_src = 'median.cl'



logging.basicConfig(level=logging.INFO)

#Kernel function instantiation
kernel = str()
with open(kernel_src, 'r') as content_file:
    kernel = content_file.read()
prg = cl.Program(ctx, kernel).build()

for i in range(1, frame_num+1):
	#Read in image
	img = imread(rel_path % i, flatten=True)

	#Allocate memory for variables on the device
	img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
	result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
	width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
	height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))

	# Call Kernel. Automatically takes care of block/grid distribution
	prg.medianFilter(queue, img.shape, None , img_g, result_g, width_g, height_g)
	result = np.empty_like(img)
	cl.enqueue_copy(queue, result, result_g)

	# Show the blurred image
	imsave(dest_path % i, result)
	logging.info('Iteration %d done.' % i)
