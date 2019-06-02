// qbrecog.cpp : 定义控制台应用程序的入口点。
//

#ifdef _DEBUG
#pragma comment(lib,"dlib19.15.0_debug_64bit_msvc1900.lib")
#else
#pragma comment(lib,"dlib19.15.0_release_64bit_msvc1900.lib")
#endif
#include "face_recognition_dlib.h"
#include "Sort_method.h"
#include <vector>
#include <string>
using namespace std;
#include <io.h>

typedef struct face_pics
{
	string name;
	std::vector<string> pics;
}face_pics;

int load_folder_jpg()
{
	return 0;
}

face_reco face_recognize;


int compareone(face_desc &desc)
{
	return 0;

}

int main(int argc, char** argv)
{
	//std::vector<string> folder;
	//listFiles("H:/git/dlib/qb_recognize/qbrecog/bin/face/*.*",folder, true);
	//FACE_RECOGNITION dlib;
	if (argc == 1)
	{
		cout << "Give some image files as arguments to this program." << endl;
		return 0;
	}

	cout << "processing image " << argv[1] << endl;

	std::string ff = "face";
	face_recognize.load_db_faces_prepare(ff);

	matrix<rgb_pixel> face_cap;
	//save the capture in the project directory
	load_image(face_cap, argv[1]);
	image_window win(face_cap);
	//Display the raw image on the screen
	//image_window win1(face_cap);

	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<matrix<rgb_pixel>> vect_faces;

	for (auto face : detector(face_cap))
	{
		auto shape = face_recognize.sp(face_cap, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(face_cap, get_face_chip_details(shape, 150, 0.25), face_chip);
		vect_faces.push_back(move(face_chip));
		//win1.add_overlay(face);
	}

	if (vect_faces.size() != 1)
	{
		cout << "Capture face error! face number " << vect_faces.size() << endl;
		return -1;
	}

	//Use DNN and get the capture face's feature with 128D vector
	std::vector<matrix<float, 0, 1>> face_cap_desc = face_recognize.net(vect_faces);
	//Browse the face feature from the database, and find the match one
	std::pair<double, std::string> candidate_face;
	std::vector<double> len_vec;

	std::vector<std::pair<double, std::string>> candi_face_vec;
	candi_face_vec.reserve(256);
	cout << "face find " << face_recognize.face_desc_vec.size() << endl;
	for (size_t i = 0; i < face_recognize.face_desc_vec.size(); ++i)
	{
		auto len = length(face_cap_desc[0] - face_recognize.face_desc_vec[i].face_feature);
		cout << "len is " << len << endl;
		if (len < 0.4)
		{
			len_vec.push_back(len);
			candidate_face.first = len;
			candidate_face.second = face_recognize.face_desc_vec[i].name.c_str();
			candi_face_vec.push_back(candidate_face);

//#ifdef _FACE_RECOGNIZE_DEBUG
//			char buffer[256] = { 0 };
//			sprintf_s(buffer, "Candidate face %s Euclid length %f",
//				face_recognize.face_desc_vec[i].name.c_str(),
//				len);
//			MessageBox(CString(buffer), NULL, MB_YESNO);
//#endif
		}
		else
		{
			cout << "This face from database is not match the face, continue!" << endl;
		}
	}

	//Find the most similar face
	if (len_vec.size() != 0)
	{
		shellSort(len_vec);

		int i(0);
		for (i = 0; i != len_vec.size(); i++)
		{
			if (len_vec[0] == candi_face_vec[i].first)
				break;
		}

		char buffer[256] = { 0 };
		sprintf_s(buffer, "The face is %s -- Euclid length %f",
			candi_face_vec[i].second.c_str(), candi_face_vec[i].first);
		cout << buffer << endl;
		/*if (MessageBox(CString(buffer), NULL, MB_YESNO) == IDNO)
		{
			face_record();
		}*/
	}
	else
	{
		cout << "not found" << endl;
		/*if (MessageBox(CString("Not the similar face been found"), NULL, MB_YESNO) == IDYES)
		{
			face_record();
		}*/
	}
	//char a;
	getchar();
    return 0;
}

