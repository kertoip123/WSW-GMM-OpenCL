/*
 * Performance.h
 *
 *  Created on: 07.01.2017
 *      Author: Piotr Janus
 */

#ifndef SRC_PERFORMANCE_HPP_
#define SRC_PERFORMANCE_HPP_

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

/**
 * Klasa obliczajaca wskazniki jakosci
 */
class Performance {
	private:
		uint64_t metrics[4];

	public:
		/**
		 * Konstruktor
		 */
		Performance();
		/**
		 * Oblicza wskazniki jakosci dla danej ramki
		 * \param result - ramka dla ktorej maja zostac obliczone wskazniki jakosc
		 * \param gt_frame - ramka wzorcowa
		 * \return void
		 */
		void count_coefficients(Mat & result, Mat & gt_frame);

		/**
		 * Wyswietla obliczne wskazniki jakosci
		 * \return void
		 */
		void print_metrics();

};

#endif /* SRC_PERFORMANCE_HPP_ */
