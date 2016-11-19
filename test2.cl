__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;


__kernel void test(
	__read_only image2d_t frame,
	__write_only image2d_t dst,
	__global float* array) 
{
	const int2 gid = { get_global_id(0), get_global_id(1) };
	const int2 size = { get_image_width(frame), get_image_height(frame) };
	
	if (!all(gid < size))
		return;
		
	float4 input = read_imagef(frame, smp, gid);
	float pix = (array[0]*input.x + array[1] * input.y + array[2] *input.z)/(array[0] + array[1] + array[2]) * 255.0f;
	float out = 0.0f;
	if (pix > 128.0f)
	{
		out = 1.0f;
	}
	
	float4 pix_out = {out, out, out, 1.0f};
	write_imagef(dst, gid, pix_out);

}