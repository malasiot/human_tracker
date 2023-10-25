#include <htrac/pose/keypoint_detector_openpose.hpp>

#include <openpose/headers.hpp>

static const std::map<std::string, std::string> coco2cmu = {
    { "LAnkle", "LeftFoot" },
    { "RAnkle", "RightFoot" },
    { "LElbow", "LeftForeArm" },
    { "RElbow", "RightForeArm" },
    { "LEye", "eye.L" },
    { "REye", "eye.R" },
    { "LHip", "LeftUpLeg" },
    { "RHip", "RightUpLeg" },
    { "LKnee", "LeftLeg" },
    { "RKnee", "RightLeg" },
    { "LShoulder", "LeftArm" },
    { "RShoulder", "RightArm" },
    { "LWrist", "LeftHand" },
    { "RWrist", "RightHand" },
    { "Neck", "Neck" }

};

static std::string coco_keypoint_to_cmu_joint(const std::string &kp) {
    auto it = coco2cmu.find(kp) ;
    if ( it == coco2cmu.end() ) return std::string() ;
    else return (*it).second ;
}

class OpenPoseAsync {

public:

    OpenPoseAsync(): wrapper_( op::ThreadManagerMode::Asynchronous ) {}
    bool config(const KeyPointDetectorOpenPose::Parameters &params) {
        try {
            const auto outputSize = op::Point(-1, -1) ;
            const auto netInputSize = op::Point(256, 256);
            const auto poseMode = op::PoseMode::Enabled;
            const auto poseModel = op::PoseModel::COCO_18;
            const auto keypointScaleMode = op::ScaleMode::InputResolution ;
            const std::vector<op::HeatMapType> heatMapTypes = { op::HeatMapType::Parts, op::HeatMapType::Background };
            const auto heatMapScaleMode = op::ScaleMode::ZeroToOne ;

            // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
            const op::WrapperStructPose wrapperStructPose{
                poseMode, netInputSize, 1.0, outputSize, keypointScaleMode,
                        -1, // num GPUs (all)
                        0, // GPU start
                        1, // scale_number
                        0.25f, // scale_gap
                        op::RenderMode::None, // skeleton render
                        poseModel,
                        true, // blending
                        0.f, // alpha_pose,
                        0.f, // alpha_heatmap
                        0, // part to show
                        op::String(params.data_folder_),
                        heatMapTypes, heatMapScaleMode,
                        false, // part candidates
                        0.05f, // render threshold for detected keypoints
                        1, // number people max,
                        false, // maximize_positives,
                        -1, // FLAGS_fps_max,
                        "", // op::String(FLAGS_prototxt_path),
                        "", //op::String(FLAGS_caffemodel_path),
                        0.f, //(float)FLAGS_upsampling_ratio, enableGoogleLogging};
                        true
            };

            wrapper_.configure(wrapperStructPose);

            wrapper_.configure(op::WrapperStructFace{});
            wrapper_.configure(op::WrapperStructHand{});
            wrapper_.configure(op::WrapperStructExtra{});

            wrapper_.start() ;

            return true ;
        }
        catch (const std::exception&e) {
            throw KeyPointDetectorException(e.what()) ;
            return false ;
        }


    }

    KeyPoints run(const cv::Mat &im) {
        try {
            // Push frame
            auto datumToProcess = createDatum(im);
            if (datumToProcess != nullptr)
            {
                auto successfullyEmplaced = wrapper_.waitAndEmplace(datumToProcess);
                // Pop frame
                std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
                if (successfullyEmplaced && wrapper_.waitAndPop(datumProcessed))
                {
                    KeyPoints kpts = getKeyPoints(datumProcessed) ;
                    return kpts ;
                }
                else {
                    op::opLog("Processed datum could not be emplaced.", op::Priority::High);

                }
            }

        } catch ( std::exception &e ) {
            throw KeyPointDetectorException(e.what()) ;
            return {} ;
        }

        return {} ;
    }

private:

    KeyPoints getKeyPoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr) const {
        try {
            KeyPoints kpts ;
            const auto bpmap = op::getPoseBodyPartMapping(op::PoseModel::COCO_18);

            if (datumsPtr != nullptr && !datumsPtr->empty()) {

                const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;

                if ( poseKeypoints.empty() ) return kpts ;

                for( uint i=0 ; i<18 ; i++ ) {

                    auto it = bpmap.find(i) ;

                    float x = poseKeypoints[3*i] ;
                    float y = poseKeypoints[3*i+1] ;
                    float w = poseKeypoints[3*i+2] ;

                    std::string jointname = coco_keypoint_to_cmu_joint(it->second) ;

                    if ( !jointname.empty() )
                        kpts.emplace(jointname, std::make_pair(Eigen::Vector2f{x, y}, w)) ;

                }

#if 0
                op::opLog("Person pose keypoints:");
                for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
                {
                    op::opLog("Person " + std::to_string(person) + " (x, y, score):");
                    for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
                    {
                        std::string valueToPrint;
                        for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
                        {
                            valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
                        }
                        op::opLog(valueToPrint);
                    }
                }
                op::opLog(" ");
                // Alternative: just getting std::string equivalent
                op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString());
                op::opLog("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString());
                op::opLog("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString());
                // Heatmaps
                const auto& poseHeatMaps = datumsPtr->at(0)->poseHeatMaps;
                if (!poseHeatMaps.empty())
                {
                    op::opLog("Pose heatmaps size: [" + std::to_string(poseHeatMaps.getSize(0)) + ", "
                              + std::to_string(poseHeatMaps.getSize(1)) + ", "
                              + std::to_string(poseHeatMaps.getSize(2)) + "]");
                    const auto& faceHeatMaps = datumsPtr->at(0)->faceHeatMaps;
                    op::opLog("Face heatmaps size: [" + std::to_string(faceHeatMaps.getSize(0)) + ", "
                              + std::to_string(faceHeatMaps.getSize(1)) + ", "
                              + std::to_string(faceHeatMaps.getSize(2)) + ", "
                              + std::to_string(faceHeatMaps.getSize(3)) + "]");
                    const auto& handHeatMaps = datumsPtr->at(0)->handHeatMaps;
                    op::opLog("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
                            + std::to_string(handHeatMaps[0].getSize(1)) + ", "
                            + std::to_string(handHeatMaps[0].getSize(2)) + ", "
                            + std::to_string(handHeatMaps[0].getSize(3)) + "]");
                    op::opLog("Right hand heatmaps size: [" + std::to_string(handHeatMaps[1].getSize(0)) + ", "
                            + std::to_string(handHeatMaps[1].getSize(1)) + ", "
                            + std::to_string(handHeatMaps[1].getSize(2)) + ", "
                            + std::to_string(handHeatMaps[1].getSize(3)) + "]");
                }
#endif

            }

            return kpts ;


        }
        catch (const std::exception& e)
        {

            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
             throw KeyPointDetectorException(e.what()) ;
            return {};
        }
    }


    std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> createDatum(const cv::Mat &im) {
        try {

            // Create new datum
            auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
            datumsPtr->emplace_back();
            auto& datumPtr = datumsPtr->at(0);
            datumPtr = std::make_shared<op::Datum>();

            // Fill datum
            datumPtr->cvInputData = OP_CV2OPCONSTMAT(im);

            if (datumPtr->cvInputData.empty()) {
                op::opLog("Empty frame detected", op::Priority::High);
                datumsPtr = nullptr;
            }

            return datumsPtr;

        }
        catch (const std::exception& e)
        {

            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    op::Wrapper wrapper_ ;

};


KeyPointDetectorOpenPose::KeyPointDetectorOpenPose(const Parameters &params): params_(params) {
    impl_.reset(new OpenPoseAsync) ;
}

KeyPointDetectorOpenPose::~KeyPointDetectorOpenPose() {

}

void KeyPointDetectorOpenPose::init() {
    impl_->config(params_) ;
}

KeyPoints KeyPointDetectorOpenPose::findKeyPoints(const cv::Mat &img) {
    return impl_->run(img) ;
}


