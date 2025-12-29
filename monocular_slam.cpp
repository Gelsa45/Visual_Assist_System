/**
* This file is part of UCOSLAM
*
* Copyright (C) 2018 Rafael Munoz Salinas
*
* GPL v3
*/

#include <cmath>
#include <string>
#include <iostream>

#include "ucoslam.h"
#include "basictypes/debug.h"
#include "mapviewer.h"
#include "basictypes/timers.h"
#include "map.h"
#include "inputreader.h"

using namespace std;

class CmdLineParser {
    int argc; char **argv;
public:
    CmdLineParser(int _argc, char **_argv):argc(_argc),argv(_argv){}
    bool operator[](string param){
        for(int i=0;i<argc;i++) if(string(argv[i])==param) return true;
        return false;
    }
    string operator()(string param,string defvalue=""){
        for(int i=0;i<argc-1;i++)
            if(string(argv[i])==param) return argv[i+1];
        return defvalue;
    }
};

int main(int argc,char **argv){
try{
    CmdLineParser cml(argc, argv);

    if(argc < 3 || cml["-h"]){
        cerr << "Usage: video camera.yml [-debug 1]" << endl;
        return -1;
    }

    InputReader vcap;
    vcap.open(argv[1], true);
    if(!vcap.isOpened())
        throw runtime_error("Video not opened");

    ucoslam::UcoSlam Slam;
    Slam.setDebugLevel(stoi(cml("-debug","0")));

    ucoslam::ImageParams image_params;
    image_params.readFromXMLFile(argv[2]);

    ucoslam::Params params;
    auto TheMap = make_shared<ucoslam::Map>();
    Slam.setParams(TheMap, params, "");

    ucoslam::MapViewer Viewer;

    cv::Mat image, camPose_c2g;

    // ========= YAW STATE (ONLY ONCE) =========
    static bool yaw_initialized = false;
    static double prev_yaw_deg = 0.0;
    static double yaw_accum = 0.0;
    // =======================================

    while(true){
        vcap >> image;
        if(image.empty()) break;

        camPose_c2g = Slam.process(image, image_params, vcap.getCurrentFrameIndex());
        std::cout << "Frame processed" << std::endl;

        // ================= SLAM YAW DIRECTION =================
        if(camPose_c2g.rows == 4 && camPose_c2g.cols == 4){


            cv::Mat R = camPose_c2g(cv::Range(0,3), cv::Range(0,3));
            R.convertTo(R, CV_64F);

            double yaw = atan2(R.at<double>(1,0), R.at<double>(0,0));
            double yaw_deg = yaw * 180.0 / CV_PI;

            if(!yaw_initialized){
                prev_yaw_deg = yaw_deg;
                yaw_initialized = true;
            }
            else{
                double delta_yaw = yaw_deg - prev_yaw_deg;
                prev_yaw_deg = yaw_deg;

                yaw_accum += delta_yaw;

                string state = "MOVING STRAIGHT";
                if(yaw_accum > 1.0){
                    state = "TURNING RIGHT";
                    yaw_accum = 0.0;
                }
                else if(yaw_accum < -1.0){
                    state = "TURNING LEFT";
                    yaw_accum = 0.0;
                }

                cout << "[SLAM] " << state << endl;
            }
        }
        // =====================================================

        Viewer.show(TheMap, image, camPose_c2g, "", Slam.getCurrentKeyFrameIndex());
    }
}
catch(exception &ex){
    cerr << ex.what() << endl;
}
}
