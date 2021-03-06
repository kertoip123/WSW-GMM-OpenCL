__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;

typedef struct MogParams
{
	float varThreshold;
	float backgroundRatio;
	//float w0; // waga dla nowej mikstury
	float var0; // wariancja dla nowej mikstury
	float minVar; // dolny prog mozliwej wariancji
} MogParams;

#ifndef nmixtures 
#define nmixtures 5
#endif

__kernel void mog_image(
	__read_only image2d_t frame,
	__write_only image2d_t dst,
	__global float* mixtureData,
	__constant MogParams* params,
	const float alpha) // krzywa uczenia
{
	const int2 gid = { get_global_id(0), get_global_id(1) };
	const int2 size = { get_image_width(frame), get_image_height(frame) };
	
	if (!all(gid < size))
		return;
	
	float4 input = read_imagef(frame, smp, gid);
	// rgb2grayscale
	float pix = (input.x + input.y + input.z)/3.0f * 255.0f;
	const int gid1 = gid.x + gid.y * size.x;
	const int size1 = size.x * size.y;
	int pdfMatched = -1;

	__private float weight[nmixtures];
	__private float mean[nmixtures];
	__private float var[nmixtures];
	__private float sortKey[nmixtures];

	#pragma unroll nmixtures
	for(int mx = 0; mx < nmixtures; ++mx)
	{
		weight[mx] = mixtureData[gid1 + size1 * (mx + 0 * nmixtures)];
		mean[mx]   = mixtureData[gid1 + size1 * (mx + 1 * nmixtures)];
		var[mx]    = mixtureData[gid1 + size1 * (mx + 2 * nmixtures)];

		if(pdfMatched < 0)
		{
			float diff = pix - mean[mx];
			float d2 = native_sqrt(diff*diff);
            //float d2 = diff*diff;
			float threshold = params->varThreshold * var[mx];
		
			// Mahalanobis distance
			if(d2 < threshold)
				pdfMatched = mx;
		}
	}
	
	if(pdfMatched < 0)
	{
		// No matching mixture found - replace the weakest one
		pdfMatched = nmixtures - 1; 

		//weight[pdfMatched] = params->w0;
		mean[pdfMatched] = pix;

       
        #pragma unroll nmixtures
		for(int mx = 1; mx < nmixtures-1; ++mx)
		{
            if (var[mx] > var[pdfMatched])
            {
                var[pdfMatched] = var[mx];
            }
        }
		var[pdfMatched] = params->var0;
	}
	else
	{
		#pragma unroll nmixtures
		for(int mx = 0; mx < nmixtures; ++mx)
		{
			if(mx == pdfMatched)
			{
				float diff = pix - mean[mx];

				#define PI_MULT_2 6.28318530717958647692f
				//#define PI_MULT_2 247.673152
                float rho = alpha / native_sqrt(PI_MULT_2 * var[mx]) * native_exp(-0.5f * diff*diff / var[mx]);

				weight[mx] = weight[mx] + alpha * (1 - weight[mx]);
				mean[mx] = mean[mx] + rho * diff;
				var[mx] = max(params->minVar, var[mx] + rho * (diff*diff - var[mx]));
			}
			else
			{
				// For the unmatched mixtures, mean and variance
				// are unchanged, only the weight is replaced by:
				// weight = (1 - alpha) * weight;

				weight[mx] = (1 - alpha) * weight[mx];
			}
		}
	}

	// Normalize weight and calculate sortKey
	float weightSum = 0.0f;
	#pragma unroll nmixtures
	for(int mx = 0; mx < nmixtures; ++mx)
		weightSum += weight[mx];

	float invSum = 1.0f / weightSum;
	#pragma unroll nmixtures
	for(int mx = 0; mx < nmixtures; ++mx)
	{
		//weight[mx] *= invSum;
		sortKey[mx] = var[mx] > FLT_MIN
			? weight[mx] / native_sqrt(var[mx])
			: 0;
	}

	// Sort mixtures (buble sort).
	// Every mixtures but the one with "completely new" weight and variance
	// are already sorted thus we need to reorder only that single mixture.
	for(int mx = 0; mx < pdfMatched; ++mx)
	{
		if(sortKey[pdfMatched] > sortKey[mx])
		{
			float weightTemp = weight[pdfMatched];
			float meanTemp = mean[pdfMatched];
			float varTemp = var[pdfMatched];

			weight[pdfMatched] = weight[mx];
			mean[pdfMatched] = mean[mx];
			var[pdfMatched] = var[mx];

			weight[mx] = weightTemp;
			mean[mx] = meanTemp;
			var[mx] = varTemp;
			break;
		}
	}

	#pragma unroll nmixtures
	for(int mx = 0; mx < nmixtures; ++mx)
	{
		mixtureData[gid1 + size1 * (mx + 0 * nmixtures)] = weight[mx];
		mixtureData[gid1 + size1 * (mx + 1 * nmixtures)] = mean[mx];
		mixtureData[gid1 + size1 * (mx + 2 * nmixtures)] = var[mx];
	}

	// No match is found with any of the K Gaussians.
	// In this case, the pixel is classified as foreground
	if(pdfMatched < 0)
	{
		float pix = 1.0f;
		write_imagef(dst, gid, (float4) {pix, pix, pix, 1.0});
		return;
	}

	// If the Gaussian distribution is classified as a background one,
	// the pixel is classified as background,
	// otherwise pixel represents the foreground
	weightSum = 0.0f;
	for(int mx = 0; mx < nmixtures; ++mx)
	{
		// The first Gaussian distributions which exceed
		// certain threshold (backgroundRatio) are retained for 
		// a background distribution.

		// The other distributions are considered
		// to represent a foreground distribution
		weightSum += weight[mx];

		if(weightSum > params->backgroundRatio)
		{
			float pix = pdfMatched > mx 
				? 1.0f // foreground
				: 0.0f;  // background
			write_imagef(dst, gid, (float4) {pix, pix, pix, 1.0});
			return;
		}
	}
}