__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;


__kernel void performance(
	__read_only image2d_t out_frame,
	__read_only image2d_t gt_frame,
	__global int* performance_data) 
{
	const int2 gid = { get_global_id(0), get_global_id(1) };
	const int2 size = { get_image_width(out_frame), get_image_height(out_frame) };
	
	if (!all(gid < size))
		return;
	
	float4 out = read_imagef(out_frame, smp, gid);
    float4 gt = read_imagef(gt_frame, smp, gid);
	int in_pix = out.x*255;
    int gt_pix = gt.x*255;
    
    const int gid1 = gid.x + gid.y * size.x;
	const int size1 = size.x * size.y;
    
    if (gt_pix == 85)
        return;
    
    if (gt_pix == 0)
    {
        if (in_pix == 0)
        {
            // TN
            performance_data[gid1 + size1] += 1;
        }
        else
        {
            // FP
            performance_data[gid1 + 3*size1] += 1;
        }
    }
    else
    {
        if (in_pix > 0)
        {
            // TP
            performance_data[gid1] += 1;
        }
        else
        {
            // FN
            performance_data[gid1 + 2*size1] += 1;
        }
    }
}