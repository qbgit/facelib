#include "io.h"
#include "Face_recognition_dlib.h"

using namespace dlib;
using namespace std;


// ----------------------------------------------------------------------------------------



//人名的文件夹
void face_reco::listfiles(std::string dir,std::string namefolder)
{
	intptr_t handle;
	_finddata_t findData;

	string dirfilter =dir + "/" +  namefolder + "/" + "*.*";
	handle = _findfirst(dirfilter.c_str(), &findData);    // 查找目录中的第一个文件
	if (handle == -1)
	{
		cout << "Failed to find first file!\n";
		return;
	}
	face_desc desc;
	desc.name = namefolder;//子文件夹的名称为人名
	do
	{

		if (!(findData.attrib & _A_SUBDIR))
			desc.files.push_back(findData.name);
	} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件
													//cout << "Done!\n";
	_findclose(handle);    // 关闭搜索句柄
	face_desc_vec.push_back(desc);

}


void face_reco::listfolder(std::string dir, std::vector<string> & names)
{
	intptr_t handle;
	_finddata_t findData;
	
	string dirfilter = dir + "/" + "*.*";
	handle = _findfirst(dirfilter.c_str(), &findData);    // 查找目录中的第一个文件
	if (handle == -1)
	{
		cout << "Failed to find first file!\n";
		return;
	}

	do
	{
		if (findData.attrib & _A_SUBDIR)
			//&& strcmp(findData.name, ".") != 0
			//&& strcmp(findData.name, "..") != 0
			//)    // 是否是子目录并且不为"."或".."
		{
			if (findData.name[0] != '.')
				names.push_back(findData.name);
			//cout << findData.name << "\t<dir>\n";
		}
		else //否则是文件
		{
			//子文件
			//face_desc_vec.files.push_back(findData.name);
		}
		//cout << findData.name << "\t" << findData.size << endl;
	} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

													//cout << "Done!\n";
	_findclose(handle);    // 关闭搜索句柄



}



face_reco::face_reco(int depth)
{
	face_desc_vec.reserve(depth);
}

//查找文件夹下面的所有文件夹

int face_reco::load_db_faces_prepare(string folder)
{
	
	// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
	deserialize("./shape_predictor_68_face_landmarks.dat") >> sp;

	// And finally we load the DNN responsible for face recognition.
	deserialize("./dlib_face_recognition_resnet_model_v1.dat") >> net;

	std::vector<string> vecs;
	listfolder(folder.c_str(), vecs);
	auto iter = vecs.begin();
	while(iter != vecs.end())
	{
		//文件夹名称就是人名
		string name = *iter;
		cout << name << endl;
		//string mfolder = folder + "/" + name;

		listfiles(folder,name);
		iter++;
	}

	//把所有的文件都放到数据结构中，开始读取并且获取值
	load_faces(folder);


	return 0;
}



void face_reco::load_faces(string folder)
{
	frontal_face_detector detector = get_frontal_face_detector();
	auto iter = face_desc_vec.begin();
	while (iter != face_desc_vec.end())
	{
		face_desc &desc = *iter;
		matrix<rgb_pixel> img;
		string path = folder + "/" + desc.name +"/"+ desc.files[0];
		cout <<"path is " << path << endl;
		//char path[260];
		//sprintf_s(path, "%s/%s/%s", folder.c_str(),desc.name.c_str(),desc.files[0].c_str());
		load_image(img, path);
		//image_window win(img);

		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

			//Record the all this face's information
			//FACE_DESC sigle_face;
			desc.face_chip = face_chip;
			//sigle_face.name = fileinfo.name;

			std::vector<matrix<rgb_pixel>> face_chip_vec;
			std::vector<matrix<float, 0, 1>> face_all;

			face_chip_vec.push_back(move(face_chip));

			//Asks the DNN to convert each face image in faces into a 128D vector
			face_all = net(face_chip_vec);

			//Get the feature of this person
			std::vector<matrix<float, 0, 1>>::iterator iter_begin = face_all.begin(),
				iter_end = face_all.end();
			if (face_all.size() > 1) break;
			desc.face_feature = *iter_begin;

			//all the person description into vector
			//face_desc_vec.push_back(sigle_face);
		}
		iter++;
	}

}





