__constant sampler_t smp = 
    CLK_NORMALIZED_COORDS_FALSE | 
    CLK_FILTER_NEAREST | 
    CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void in_2_out( __read_only image2d_t frame, __write_only image2d_t dst)
{
    const int2 gid = { get_global_id(0), get_global_id(1) };

    write_imagef(dst, gid, (float4)read_imagef(frame, smp, gid));

    
}