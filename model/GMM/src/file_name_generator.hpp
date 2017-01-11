#ifndef SRC_FILE_NAME_GENERATOR_HPP_
#define SRC_FILE_NAME_GENERATOR_HPP_

#include <string>
using namespace std;

const string JPG = ".jpg";
const string PNG = ".png";

/**
 * Klasa odpowiadajaca za generowanie sciezek do kolejnych obrazow z danej sekwencji testowej.
 */
class FileNameGenerator
{
	private:
		string file_name_prefix;
		string file_extension;
	public:
		/**
		 * Konstruktor
		 * \param file_name_prefix - nazwa
		 * \param file_name_prefix - nazwa sekwencji wejsciowej
		 * \param file_extension - rozszerzenie obrazow w sekwencji testowej
		 */
		FileNameGenerator(string file_name_prefix, string file_extension);
		/**
		 * Generuje nazwe obrazu o danym id
		 * \param int_frame_id - id ramki ktorej nazwa ma zostac wygenerowana
		 * \return nazwa ramki
		 */
		string get_frame_name(int frame_id);
};

#endif /* SRC_FILE_NAME_GENERATOR_HPP_ */
