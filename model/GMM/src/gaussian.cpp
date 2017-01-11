#include "gaussian.hpp"
#include "tools.hpp"
#include <iostream>
using namespace std;

double Gaussian::alpha = 0;

//#define THRESHOLD (5)
#define THRESHOLD (2.5)

void Gaussian::initialise(double weight, double *gaussian_mean, double standard_deviation)
{
    rgb_mean = new double[RGB_COMPONENTS_NUM];
    this->weight = weight;
    for(int i = 0; i < RGB_COMPONENTS_NUM; ++i)
    {
        rgb_mean[i] = gaussian_mean[i];
    }
    this->standard_deviation = standard_deviation;
}

void Gaussian::get_rgb_mean(double * gaussian_means)
{
    for(int i = 0; i < RGB_COMPONENTS_NUM; ++i)
    {
        gaussian_means[i]= rgb_mean[i];
    }
}

void Gaussian::print()
{
    string delimiter = ", ";
    cout<<"("<<weight<<delimiter<<rgb_mean[0]<<delimiter;
    cout<<rgb_mean[1]<<delimiter<<rgb_mean[2]<<delimiter;
    cout<<standard_deviation<<")";
}

bool Gaussian :: operator<(const Gaussian& gaussian) const
{
    return (get_sort_parameter() > gaussian.get_sort_parameter());
}

double Gaussian::get_sort_parameter() const
{
    return (weight/standard_deviation);
}

void Gaussian::update_unmatched()
{
    weight *= (1 - alpha);
}

void Gaussian::update_matched(double *rgb)
{
    //Weight
    weight *=  (1 - alpha);
    weight += alpha;

    //RGB means
    double ro = alpha*count_probability_density(rgb, rgb_mean, standard_deviation);
    //double ro = alpha;
    for(int i = 0; i < RGB_COMPONENTS_NUM; ++i)
    {
        rgb_mean[i] *= (1 - ro);
        rgb_mean[i] += ro*rgb[i];
    }/**/

    //Standard deviation
    double dist = malahidanDistance(rgb, rgb_mean, RGB_COMPONENTS_NUM);
    standard_deviation *= standard_deviation;
    standard_deviation *= (1 - ro);
    standard_deviation += ro*dist;
    standard_deviation = sqrt(standard_deviation);/**/
}

bool Gaussian::check_pixel_match(double *rgb) //sqrt((r-r_mean)^2 + (b-b_mean)^2 + (c-c_mean)^2)  / dev
{
    double dist = malahidanDistance(rgb, rgb_mean, RGB_COMPONENTS_NUM);
    dist = sqrt(dist);
    dist /= standard_deviation;
    double threshold = THRESHOLD * standard_deviation;
    if(dist < threshold)
        return true;
    else
        return false;
}

