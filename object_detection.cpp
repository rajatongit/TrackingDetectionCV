#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
//#include <nms.hpp>
//#include <utils.hpp>

void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);
#define USING_WINDOWS
// #define USING_LINUX
//#define DEBUG_FLAG
// #define DEBUG_FLAG2
#define TRAIN 1
#define TEST 0
//#define TASK3 2
//#define TASK3TEST 3

void loadImages(const int& classNumber, const int& imageNumber, std::vector<cv::Mat>& imgList, int train_test) {
	char* trainFolder = new char[128];
#ifdef USING_WINDOWS
	if (train_test == 1) { // train = 1
		strcpy(trainFolder, "D:\\Exercise030405\\data\\task3\\train");
		sprintf(trainFolder, "%s\\%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s\\%04d.jpg", trainFolder, imageNumber);
	} else {
		strcpy(trainFolder, "D:\\Exercise030405\\data\\task3\\test");
		//sprintf(trainFolder, "%s\\%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s\\%04d.jpg", trainFolder, imageNumber);
	}
#endif
#ifdef USING_LINUX
	if (train_test == 1) { //train = 1
		strcpy(trainFolder, "/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task3/train");
		sprintf(trainFolder, "%s/%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s/%04d.jpg", trainFolder, imageNumber);
	} else {
		strcpy(trainFolder, "/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task3/train");
		//sprintf(trainFolder, "%s/%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s/%04d.jpg", trainFolder, imageNumber);
	}

#endif
    cv::Mat img = cv::imread (trainFolder); // load the image
    if(!img.data) {
        std::cout << "Could not load the image..." << std::endl;
    }
#ifdef DEBUG_FLAG
    std::cout << "debugging mode : ON" << std::endl;
    cv::imshow("test", img);
    cv::waitKey(5000); // show image for 5000ms
#endif
    imgList.push_back(img.clone());
}

void computeHOG(const std::vector<cv::Mat> &imgList, const cv::Size& HOGSize, std::vector<cv::Mat> &descriptorsList) {
    int index = 0;
    cv::HOGDescriptor hogDetector;
    hogDetector.winSize = HOGSize;
    for (std::vector<cv::Mat>::const_iterator it = imgList.begin(); it != imgList.end() ; ++it ) {
        cv::Mat img = (*it).clone();
        cv::resize(img, img, cv::Size(64, 64));
        std::vector<cv::Point> locations;
        std::vector<float> descriptors;
        cv::Mat grayImg;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        hogDetector.compute(grayImg, descriptors, cv::Size(8, 8), cv::Size(0, 0), locations);
        descriptorsList.push_back((cv::Mat(descriptors)).clone());
#ifdef DEBUG_FLAG2
        std::cout << "debugging mode : ON " << index++ << std::endl;
        cv::imshow("test", img);
        visualizeHOG(img, descriptors, hogDetector, 10);
        cv::waitKey(5000); // show image for 5000ms
#endif
        descriptors.clear(); // ASK: Maybe we don't need to clear
        locations.clear(); // ASK: Maybe we don't need to clear
    }
}
void alignData(const std::vector<cv::Mat>& descriptorsList, cv::Mat trainingData) {
    const int rows = (int)descriptorsList.size();
	//std::cout << "descriptors list size is " << descriptorsList.size() << " rows, cols " << descriptorsList[0].rows << " " <<
		//descriptorsList[0].cols << std::endl;
	//const int cols = (int)descriptorsList[0].cols; 
	const int cols = (int)std::max( descriptorsList[0].cols, descriptorsList[0].rows );
    cv::Mat temp(1, cols, CV_32FC1);
    int index = 0;
    for (std::vector<cv::Mat>::const_iterator it = descriptorsList.begin(); it != descriptorsList.end(); ++it, ++index) {
		std::cout << index << std::endl;
        CV_Assert( it->cols == 1 || it->rows == 1);
        if (it->cols == 1) {
			//std::cout << "cols is 1" << std::endl;
            transpose(*(it), temp);
            temp.copyTo(trainingData.row(index));
			std::cout << "training data size " << trainingData.size << std::endl;
        }
        else if (it->rows == 1) {
			//std::cout << "rows is 1" << std::endl;
            it->copyTo(trainingData.row(index));
        }
    }
}


void randomForest(const std::vector<cv::Mat>& descriptorsList, const std::vector<int>& labels, const int& task_number) {
	int iterations = 100;
	double epsilon = 0.01f;
	const int rows = (int)descriptorsList.size();
	const int cols = (int)std::max(descriptorsList[0].cols, descriptorsList[0].rows);//rows; //let's try this // HACK
	cv::Mat trainingData = cv::Mat(rows, cols, CV_32FC1);
	alignData(descriptorsList, trainingData);
	//try {
	static cv::Ptr<cv::ml::TrainData> trainingDataForForest = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	//} catch (cv::Exception & e) {
		//std::cerr << e.msg << std::endl; // output exception message
	//}
	cv::Ptr<cv::ml::RTrees> randomTrees = cv::ml::RTrees::create();
	randomTrees->setMaxDepth(10);
	randomTrees->setMinSampleCount(10);
	randomTrees->setRegressionAccuracy(0);
	randomTrees->setUseSurrogates(false);
	randomTrees->setMaxCategories(15);
	randomTrees->setCalculateVarImportance(true);
	randomTrees->setActiveVarCount(4);
	randomTrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + (epsilon > 0 ? cv::TermCriteria::EPS : 0), iterations, epsilon));
	//randomTrees->setCVFolds(0);
	//randomTrees->setUse1SERule(false);
	//randomTrees->setTruncatePrunedTree(false);
	randomTrees->setPriors(cv::Mat());
	randomTrees->train(trainingDataForForest);
	randomTrees->save("trainedRFTask3.yml");
}
void randomForestTest(const std::vector <cv::Mat> &descriptorsListTest, const std::vector<int> &labels) {
	float accuracy;
	const int rows = (int)descriptorsListTest.size();
	//std::cout << "number of iterations " << rows << std::endl;
	const int cols = (int)std::max(descriptorsListTest[0].cols, descriptorsListTest[0].rows);//rows; //let's try this // HACK
	cv::Mat testingData = cv::Mat(rows, cols, CV_32FC1);
	alignData(descriptorsListTest, testingData);
	cv::Ptr<cv::ml::RTrees> model = cv::ml::StatModel::load<cv::ml::RTrees>("trainingRF.yml");
	if (model.empty()) {
		std::cout << "couldn't load the model" << std::endl;
	}
	for (int index = 0; index < testingData.rows; index++) {
		float output = model->predict(testingData.row(index));
		int groundTruth = labels[index];
		int prediction = output;
		std::cout << "groundTruth : " << labels[index] << "prediction : " << prediction << std::endl;
		if (groundTruth == prediction) {
			accuracy++;
		}
	}
	std::cout << "Accuracy Achieved by randomForest Model: " << accuracy / testingData.rows << std::endl;
}
void slidingWindow(const cv::Mat img, const int stride, const int width, const int height, std::vector<cv::Rect> boxes) {
	int scales = 4;
	cv::Ptr<cv::ml::RTrees> model = cv::ml::StatModel::load<cv::ml::RTrees>("trainedRFTask3.yml");
	if (model.empty()) {
		std::cout << "couldn't load the model" << std::endl;
	}
	std::vector<cv::Rect> allRectangles0;
	std::vector<cv::Rect> allRectangles1;
	std::vector<cv::Rect> allRectangles2;
	std::vector<float> allScores0;
	std::vector<float> allScores1;
	std::vector<float> allScores2;
	std::vector<int> allIndices0;
	std::vector<int> allIndices1;
	std::vector<int> allIndices2;
	for (float scale = 1; scale < scales;) {
		//std::cout << scale << std::endl;
		int widthWindow = (int)(width * scale);
		int heightWindow = (int)(height * scale);
		for (int rowIndex = 0; rowIndex <= img.rows - heightWindow; rowIndex += stride) {
			for (int colIndex = 0; colIndex <= img.cols - widthWindow; colIndex += stride) {
				cv::Rect windows(colIndex, rowIndex, widthWindow, heightWindow);
				cv::Mat result = img.clone();
				cv::rectangle(result, windows, cv::Scalar(255, 0, 0));
				cv::Mat crop = img(windows);
				cv::imshow("slided window", result);
				cv::imshow("crop", crop);
				cv::resize(crop, crop, cv::Size(64, 64));
				//cv::waitKey(1);

				std::vector<cv::Mat> croppedVector;
				std::vector<cv::Mat> descriptorsList;
				croppedVector.push_back(crop);
				
				computeHOG(croppedVector, cv::Size(64, 64), descriptorsList);
				const int rows = (int)descriptorsList.size();
				//std::cout << "number of iterations " << rows << std::endl;
				const int cols = (int)std::max(descriptorsList[0].cols, descriptorsList[0].rows);
				cv::Mat testingData = cv::Mat(rows, cols, CV_32FC1);
				alignData(descriptorsList, testingData);
				cv::Mat outputProb;
				float output = model->predict(testingData.row(0));
				//float prob = model->(testingData.row(0));
				//float outputVotes = model->getVotes(testingData.row(0));
				model->getVotes(testingData.row(0), outputProb, 0);
				//std::cout << outputProb.rows << outputProb.cols <<  std::endl;
				//std::cout << outputProb << std::endl;
				/*std::cout << "tooth : " << outputProb.at<int>(1, 0) << std::endl;
				std::cout << "motor : " << outputProb.at<int>(1, 1) << std::endl;
				std::cout << "blackbox : " << outputProb.at<int>(1, 2) << std::endl;
				std::cout << "background : " << outputProb.at<int>(1, 3) << std::endl;
				*/
				cv::Mat temp;
				outputProb(cv::Range(1, 2), cv::Range(0, outputProb.cols)).copyTo(temp);
				//std::cout << "temp is " << temp << std::endl;
				//Initialize m
				double minVal;
				double maxVal;
				cv::Point minLoc;
				cv::Point maxLoc;

				cv::minMaxLoc(temp, &minVal, &maxVal, &minLoc, &maxLoc);

				//std::cout << "min val : " << minVal << "min Loc " << minLoc.x << std::endl;
				//std::cout << "max val: " << maxVal << "max Loc " << maxLoc.x << std::endl;

				if (maxLoc.x == 3) { //background class
					continue;
				}
				else {
					if (maxLoc.x == 2) {
						allRectangles2.push_back(windows);
						allScores2.push_back(maxVal / 100.0f);
					} else if (maxLoc.x == 1) {
						allRectangles1.push_back(windows);
						allScores1.push_back(maxVal / 100.0f);
					} else {
						allRectangles0.push_back(windows);
						allScores0.push_back(maxVal / 100.0f);
					}
				}
				//cv::waitKey(0);
			}
		}
		scale = scale + 0.5;
	}
	int index = 0;
	cv::Mat result2 = img.clone();

	for (std::vector<cv::Rect>::const_iterator it = allRectangles0.begin(); it != allRectangles0.end(); ++it, ++index) {
		cv::rectangle(result2, *it, cv::Scalar(0, 0, 255));
	}
	for (std::vector<cv::Rect>::const_iterator it = allRectangles1.begin(); it != allRectangles1.end(); ++it, ++index) {
		cv::rectangle(result2, *it, cv::Scalar(0, 255, 0));
	}
	for (std::vector<cv::Rect>::const_iterator it = allRectangles2.begin(); it != allRectangles2.end(); ++it, ++index) {
		cv::rectangle(result2, *it, cv::Scalar(255, 0, 0));
	}
	
	try {
		cv::putText(result2, "bounding boxes", cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 0));
		cv::dnn::NMSBoxes(allRectangles0, allScores0, 0.2, 0.3, allIndices0);
		std::cout << allIndices0.size() << std::endl;	
		cv::dnn::NMSBoxes(allRectangles1, allScores1, 0.2, 0.3, allIndices1);
		std::cout << allIndices1.size() << std::endl;
		cv::dnn::NMSBoxes(allRectangles2, allScores2, 0.2, 0.3, allIndices2);
		std::cout << allIndices2.size() << std::endl;
	}
	catch (cv::Exception &e) {
		std::cerr << e.msg << std::endl; // output exception message
	}

	cv::Mat result3 = img.clone();
	if (allIndices0.size()) {
		cv::rectangle(result3, allRectangles0[allIndices0[0]], cv::Scalar(0, 0, 255));
		cv::imshow("result3", result3);
		cv::waitKey(0);
	}
	if (allIndices1.size()) {
		cv::rectangle(result3, allRectangles1[allIndices1[0]], cv::Scalar(0, 255, 0));
		cv::imshow("result3", result3);
		cv::waitKey(0);
	}
	if (allIndices2.size()) {
		cv::rectangle(result3, allRectangles2[allIndices2[0]], cv::Scalar(255, 0, 0));
		cv::imshow("result3", result3);
		cv::waitKey(0);
	}

	cv::imwrite("result3.png", result3);
	cv::imwrite("result2.png", result2);

	cv::imshow("result2", result2);
	cv::waitKey(0);
}

void randomForestPrediction(const std::vector<cv::Mat>& imgList, const cv::Size& HOGSize) {
	std::vector<cv::Rect> boundingBox;
	std::cout << imgList.size() << std::endl;
	for (int index = 0; index < imgList.size(); index++) {
		boundingBox.clear();
		slidingWindow(imgList[index], 20, 64, 64, boundingBox);
	}
}



int main() {
	printf("OpenCV: %s", cv::getBuildInformation().c_str());
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	/*TASK3*/
	int numberOfClassesTask3 = 4;
	std::vector<int> labelsTask3;
	int imageNumberTask3[] = { 52, 80, 50, 289 };
	std::vector<cv::Mat> imageListTask3Train;
	for (int classIndex = 0; classIndex < numberOfClassesTask3; classIndex++) {
		for (int imageIndex = 0; imageIndex <= imageNumberTask3[classIndex]; imageIndex++) {
			//std::cout << classIndex << imageIndex << std::endl;
			loadImages(classIndex, imageIndex, imageListTask3Train, TRAIN);
			labelsTask3.push_back(classIndex);
		}
		std::cout << "Number of images loaded till now: " << imageListTask3Train.size() << std::endl;
		std::cout << "Number of labels assigned till now: " << labelsTask3.size() << std::endl;
	}
	std::cout << "Total number of labels assigned: " << labelsTask3.size() << std::endl;
	std::cout << "Total number of images loaded: " << imageListTask3Train.size() << std::endl;
	std::vector<cv::Mat> descriptorsListTask3;
	computeHOG(imageListTask3Train, cv::Size(64, 64), descriptorsListTask3);
	//randomForest(descriptorsListTask3, labelsTask3, 3); //passing 3 for task3
	numberOfClassesTask3 = 1;
	int numberOfTestImages = 43;
	std::vector<cv::Mat> imageListTask3Test;
	for (int classIndex = 0; classIndex < numberOfClassesTask3; classIndex++) {
		for (int imageIndex = 0; imageIndex <= numberOfTestImages; imageIndex++) {
			loadImages(classIndex, imageIndex, imageListTask3Test, TEST);
		}
		std::cout << "Number of test images loaded till now: " << imageListTask3Test.size() << std::endl;
	}
	std::cout << "Total number of test images loaded: " << imageListTask3Test.size() << std::endl;
	randomForestPrediction(imageListTask3Test, cv::Size(64, 64));

	return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {

    cv::Mat visual_image;
    cv::resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));

    int n_bins = hog_detector.nbins;
    float rad_per_bin = 3.14 / (float) n_bins;
    cv::Size win_size = hog_detector.winSize;
    cv::Size cell_size = hog_detector.cellSize;
    cv::Size block_size = hog_detector.blockSize;
    cv::Size block_stride = hog_detector.blockStride;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = win_size.width / cell_size.width;
    int cells_in_y_dir = win_size.height / cell_size.height;
    int n_cells = cells_in_x_dir * cells_in_y_dir;
    int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

    int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
    int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
    int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[n_bins];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < n_bins; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }


    // compute gradient strengths per cell
    int descriptorDataIdx = 0;


    for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
        for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
            int cell_start_x = block_x * block_stride.width / cell_size.width;
            int cell_start_y = block_y * block_stride.height / cell_size.height;

            for (int cell_id_x = cell_start_x;
                 cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
                for (int cell_id_y = cell_start_y;
                     cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

                    for (int bin = 0; bin < n_bins; bin++) {
                        float val = feats.at(descriptorDataIdx++);
                        gradientStrengths[cell_id_y][cell_id_x][bin] += val;
                    }
                    cellUpdateCounter[cell_id_y][cell_id_x]++;
                }
        }
    }


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < n_bins; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cell_size.width;
            int drawY = celly * cell_size.height;

            int mx = drawX + cell_size.width / 2;
            int my = drawY + cell_size.height / 2;

            rectangle(visual_image,
                      cv::Point(drawX * scale_factor, drawY * scale_factor),
                      cv::Point((drawX + cell_size.width) * scale_factor,
                                (drawY + cell_size.height) * scale_factor),
                      CV_RGB(100, 100, 100),
                      1);

            for (int bin = 0; bin < n_bins; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * rad_per_bin + rad_per_bin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cell_size.width / 2;
                float scale = scale_factor / 5.0; // just a visual_imagealization scale,

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1 * scale_factor, y1 * scale_factor),
                     cv::Point(x2 * scale_factor, y2 * scale_factor),
                     CV_RGB(0, 0, 255),
                     1);

            }

        }
    }


    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    cv::imshow("HOG vis", visual_image);
    cv::waitKey(-1);
    cv::imwrite("hog_vis.jpg", visual_image);

}