import pyopencl as cl
import logging

NVIDIA_PLATFORM = 'NVIDIA CUDA'
INTEL_PLATFORM = 'Intel(R) OpenCL'

GPU_DEVICE = 'GPU'
CPU_DEVICE = 'CPU'

def device_choose(preffered_platform, preffered_device):
    if not preffered_device:
        preffered_device = CPU_DEVICE

    for platform in cl.get_platforms():
        logging.debug(platform)
        platform_name = platform.get_info(cl.platform_info.NAME)
        if not preffered_platform or platform_name == preffered_platform :
            for device in platform.get_devices():
                logging.debug(device)
                device_type = cl.device_type.to_string(device.type)
                if device_type == preffered_device:
                    device_name = device.get_info(cl.device_info.NAME)
                    logging.info('Platform: ' + platform_name)
                    logging.info('Device: ' + device_name)
                    return device
                    
    logging.debug('Device not found')
    return None
	
if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	
	print device_choose(NVIDIA_PLATFORM, GPU_DEVICE)
	print device_choose(INTEL_PLATFORM, GPU_DEVICE)
	print device_choose(INTEL_PLATFORM, CPU_DEVICE)