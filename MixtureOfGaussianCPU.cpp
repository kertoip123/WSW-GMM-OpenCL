#include "MixtureOfGaussianCPU.h"

#ifdef HAVE_TBB
#include <tbb/tbb.h>
#endif

MixtureOfGaussianCPU::MixtureOfGaussianCPU(int rows, int cols, int history)
	: rows(rows)
	, cols(cols)
	, nmixtures(defaultNumMixtures)
	, history(history)
	, nframe(0)
	, backgroundRatio(defaultBackgroundRatio)
	, varThreshold(defaultVarianceThreshold)
	, noiseSigma(defaultNoiseSigma)
	, initialWeight(defaultInitialWeight)
{
	// Gaussian mixtures data
	int mix_data_size = cols * rows * nmixtures;
	bgmodel.create(1, mix_data_size * sizeof(MixtureData) / sizeof(float), CV_32F);
	bgmodel = cv::Scalar::all(0);
}

void MixtureOfGaussianCPU::operator() (cv::InputArray in, cv::OutputArray out,
	float learningRate)
{
	cv::Mat frame = in.getMat();

	++nframe;
	float alpha = learningRate >= 0 && nframe > 1 
		? learningRate
		: 1.0f/std::min(nframe, history);

	out.create(frame.size(), CV_8U);
	cv::Mat mask = out.getMat();

	calc_impl(frame.data, mask.data, bgmodel.ptr<MixtureData>(), alpha);
}

void MixtureOfGaussianCPU::reinitialize(float backgroundRatio)
{
	this->backgroundRatio = backgroundRatio;
	bgmodel = cv::Scalar::all(0);
}

void MixtureOfGaussianCPU::calc_pix_impl(uchar src, uchar* dst, 
	MixtureData mptr[], float alpha)
{
	const float w0 = initialWeight; // 0.05 lub 0.001
	const float var0 = (noiseSigma*noiseSigma*4); // 900.0 lub 50.0f
	const float minVar = (noiseSigma*noiseSigma); // 225.0

	float pix = static_cast<float>(src);
	int pdfMatched = -1;

	for(int mix = 0; mix < nmixtures; ++mix)
	{
		float diff = pix - mptr[mix].mean;
		float d2 = diff*diff;
		float threshold = varThreshold * mptr[mix].var;

		// To samo co:
		//  if (diff > -2.5f * var && 
		//		diff < +2.5f * var)

		// Mahalanobis distance
		if(d2 < threshold)
		{
			pdfMatched = mix;
			break;
		}
	}

	if(pdfMatched < 0)
	{
		// No matching mixture found - replace the weakest one
		//pdfPatched = mix = std::min(mix, nmixtures-1);
		pdfMatched = nmixtures - 1; 

		// First, decrease sum of all mixture weights by old weight 
		// weightSum -= mptr[pdfMatched].weight;
		// Then, increase it by new initial weight
		// weightSum += w0;

		mptr[pdfMatched].weight = w0;
		mptr[pdfMatched].mean = pix;
		mptr[pdfMatched].var = var0;
	}
	else
	{
		for(int mix = 0; mix < nmixtures; ++mix)
		{
			float weight = mptr[mix].weight;

			if(mix == pdfMatched)
			{
				float mu = mptr[mix].mean;
				float diff = pix - mu;
				float var = mptr[mix].var;

				//static const float PI = 3.14159265358979323846f;
				//float ni = 1.0f / sqrtf(2.0f * PI * var) * expf(-0.5f * diff*diff / var);

				mptr[mix].weight = weight + alpha * (1 - weight);
				mptr[mix].mean = mu + alpha * diff;
				mptr[mix].var = std::max(minVar, var + alpha * (diff*diff - var));
			}
			else
			{
				// For the unmatched mixtures, mean and variance
				// are unchanged, only the weight is replaced by:
				// weight = (1 - alpha) * weight;

				mptr[mix].weight = (1 - alpha) * weight;
			}
		}
	}

	// Normalize weight and calculate sortKey
	float weightSum = 0.0f;
	for(int mix = 0; mix < nmixtures; ++mix)
		weightSum += mptr[mix].weight;

	float invSum = 1.0f / weightSum;
	float sortKey[5];
	for(int mix = 0; mix < nmixtures; ++mix)
	{
		mptr[mix].weight *= invSum;
		sortKey[mix] = mptr[mix].var > DBL_MIN
			? mptr[mix].weight / sqrtf(mptr[mix].var)
			: 0;
	}

	// Sort mixtures (buble sort).
	// Every mixtures but the one with "completely new" weight and variance
	// are already sorted thus we need to reorder only that single mixture.

	for(int mix = 0; mix < pdfMatched; ++mix)
	{
		if(sortKey[pdfMatched] > sortKey[mix])
		{
			swap(mptr[pdfMatched], mptr[mix]);
			break;
		}
	}

	// No match is found with any of the K Gaussians.
	// In this case, the pixel is classified as foreground
	if(pdfMatched < 0)
	{
		*dst = 255;
		return;
	}

	// If the Gaussian distribution is classified as a background one,
	// the pixel is classified as background,
	// otherwise pixel represents the foreground
	weightSum = 0.0f;
	for(int mix = 0; mix < nmixtures; ++mix)
	{
		// The first Gaussian distributions which exceed
		// certain threshold (backgroundRatio) are retained for 
		// a background distribution.

		// The other distributions are considered
		// to represent a foreground distribution
		weightSum += mptr[mix].weight;

		if(weightSum > backgroundRatio)
		{
			*dst = pdfMatched > mix 
				? 255 // foreground
				: 0;  // background
			return;
		}
	}
}

void MixtureOfGaussianCPU::calc_impl(uchar* frame, uchar* mask,
	MixtureData* mptr, float alpha)
{
#ifndef HAVE_TBB

	for(int y = 0; y < rows; ++y)
	{
		const uchar* src = &frame[y * cols];
		uchar* dst = &mask[y * cols];

		for(int x = 0; x < cols; ++x, mptr += nmixtures)
		{
			calc_pix_impl(src[x], &dst[x], mptr, alpha);
		}
	}
#else
	tbb::parallel_for(tbb::blocked_range<int>(0, rows),
		[&](const tbb::blocked_range<int>& range)
		{
			MixtureData* mptr_local = mptr + range.begin() * cols * nmixtures;

			for(int y = range.begin(); y < range.end(); ++y)
			{
				const uchar* src = &frame[y * cols];
				uchar* dst = &mask[y * cols];

				for(int x = 0; x < cols; ++x, mptr_local += nmixtures)
				{
					calc_pix_impl(src[x], &dst[x], mptr_local, alpha);
				}
			}
		});
#endif
}