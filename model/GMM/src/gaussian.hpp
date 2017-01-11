#ifndef SRC_GAUSSIAN_HPP_
#define SRC_GAUSSIAN_HPP_

#include <opencv2/core/core.hpp>
using namespace cv;

/**
 * Klasa reprezentujaca pojedynczy rozklad Gaussa
 */
class Gaussian
{
    private:
		static double alpha;

		double weight;
        double *rgb_mean;
        double standard_deviation; //same for all RGB values, sigma without square
        bool foreground;
        double get_sort_parameter() const;

    public:
        /**
         * Inicjalizuje rozklad Gaussa
         * \param weight - waga
         * \param gaussian_mean - wartosci srednie skladowych RGB
         * \param standard_deviation - odchylenie standardowe
         * \return void
         */
        void initialise(double weight, double *gaussian_mean, double standard_deviation);

        /**
         * Zapisuje do tablicy wartosci srednie skladowych RGB
         * \param gaussian_means - wskaznik do tablicy gdzie maja zostac zapisane wartosic srednie
         * \return void
         */
        void get_rgb_mean(double * gaussian_means);
        /**
         * Pobiera wage rozkladu Gaussa
         * \return void
         */
        double get_weight() { return weight; }

        /**
         * Uaktualnia rozklad Gaussa w przypadku gdy nie przeszedl on testu dopasowania
         * \return void
         */
        void update_unmatched();

        /**
         * Uaktualnia rozklad Gaussa w przypadku gdy przeszedl on testu dopasowania
         * \param rgb - wektor RGB nowego piksela
         * \return void
         */
        void update_matched(double *rgb);

        /**
         * Sprawdza czy dany piksel pasuje do rozkladu Gaussa
         * \param rgb - wektor rgb nowego piksela
         * \return true jezeli piksel pasuje, w przeciwnym razie false
         */
        bool check_pixel_match(double *rgb);

        /**
         * Przeciazenie operatora <
         * \param gaussian - rozklad Gaussa do porownania
         * \return rezultat prownania
         */
        bool operator<(const Gaussian& gaussian) const;

        /**
         * Ustawia flage okreslajaca jaki typ piksela opisuje danych rozklad
         * \param foreground - flaga do ustawiania (jezeli true rozklad reprezentuje pierwszyplan)
         * \return void
         */
        void set_isForeground(bool foreground) { this->foreground = foreground; }

        /**
         * Sprawdza jaki typ piksela opisuje danych rozklad
         * \return true jezeli rozklad reprezentuje pierwszyplan, w przeciwnym wypadku false
         */
        bool isForeground() { return foreground; }

        /**
         * Pobiera aktualne odchylenie standardowe danego rozkladu
         * \return odchylenie standardowe
         */
        double get_deviation() { return standard_deviation; }

        void print();

        /**
         * Ustawia wartosc wspolczynnika uczenia
         * \param alpha - wspolczynnik uczenia
         * \return void
         */
        static void set_alpha(double alpha) {Gaussian::alpha = alpha; }
};


#endif /* SRC_GAUSSIAN_HPP_ */
