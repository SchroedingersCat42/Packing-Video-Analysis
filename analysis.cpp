#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <boost/program_options.hpp>

/**
 *Contributors: Michael Steck, Ann-Kathrin Raab
 *June 2015 
 *Schülerforschungszentrum Ulm (SFZ)
*/
using namespace std;
using namespace cv;

namespace po = boost::program_options;

int main(int argc, char** argv){
  
  //parse arguments
  po::options_description desc("FUU");
  desc.add_options()
    ("debug", "show debug output")
    ("blur-bs", po::value<int>()->default_value(21), "set blur block size")
    ("blur-std", po::value<double>()->default_value(7.0), "set blur std")
    ("thr", po::value<double>()->default_value(170), "set threshold level")
    ("rect-e", po::value<double>()->default_value(0.025), "set approxpoly epsilon")
  ;
  po::variables_map opts;
  po::store(po::parse_command_line(argc, argv, desc), opts);
  po::notify(opts);
  

  /*
   * OpenCV Main
   */

  VideoCapture vid("../test27.mov");
  if(!vid.isOpened()){
    cout << "fuu" << endl; 
    return -1;
  }

  // window and filestream
  namedWindow("output",1);
  ofstream fs;
  fs.open("out.dat");
  
  for(int num=0;;num++){
    
    Mat frame, imageH, imageS, imageV, imageHSV, imageT, imageTmp, imageWarp;
    vector<Mat> imageCh;
    
	// get next frame
    vid >> frame;
	if(frame.empty()) break;
    
	// convert to HSV and threshold
    cvtColor(frame, imageHSV, CV_BGR2HSV);
    split(imageHSV, imageCh);
	GaussianBlur(imageCh[1], imageTmp, Size(opts["blur-bs"].as<int>(), opts["blur-bs"].as<int>()), opts["blur-std"].as<double>());
	threshold(imageTmp, imageT, opts["thr"].as<double>(), 255, THRESH_BINARY_INV);

	// find outermost contour
	Mat edges;
	vector<vector<Point> > contours;
	Canny(imageT, edges, 20, 20*3, 3);
	findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(),
		[](vector<Point> a, vector<Point> b) -> bool { return contourArea(a) > contourArea(b); }
	);
	//drawContours(frame, contours, 0, Scalar(80, 200, 120), 3);

	// fit Rect
	vector<Point> rect;
	approxPolyDP(Mat(contours[0]), rect, arcLength(contours[0], false)*opts["rect-e"].as<double>(), true);
	polylines(frame, rect, true, Scalar(120, 80, 200), 2);

	// only for real rectangle
	if (rect.size() != 4) continue;

	// unmorph rect
	vector<Point2f> rect2f;
	for (auto it=rect.begin(); it!=rect.end(); it++) {
		rect2f.push_back(Point2f((*it).x, (*it).y));
	}
	vector<Point2f> warpPoints {Point2f(0.0f,0.0f), Point2f(0.0f,560.0f), Point2f(150.0f,560.0f), Point2f(150.0f,0.0f)};
	Mat M = getPerspectiveTransform(rect2f, warpPoints);
	warpPerspective(imageCh[2], imageWarp, M, Size(150,560));

	// ROI
	Mat imageROI(imageWarp, Rect(20,20,110,520));
	Mat lines;

	// compute row average and save
	reduce(imageWarp, lines, 1, CV_REDUCE_AVG, CV_32FC1);
		fs << num << "\t";
	for( int j=0; j<lines.rows; j++ ){
		fs << lines.at<float>(j) << "\t";
	}	
		fs << endl;
		cout << num << ", " << flush;
 
	if(opts.count("debug")) {   
		imshow("output", imageROI);
		if(waitKey(1) >= 0) break;
	}
  }

  fs.close();

}
