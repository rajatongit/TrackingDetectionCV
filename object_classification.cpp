#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>

void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);
//inline TermCriteria TC(int iters, double eps); 
#define USING_WINDOWS
// #define USING_LINUX
//#define DEBUG_FLAG
// #define DEBUG_FLAG2
#define TRAIN 1
#define TEST 0
#define TASK3 2
#define TASK3TEST 3

void loadImages(const int& classNumber, const int& imageNumber, std::vector<cv::Mat>& imgList, int train_test) {
	//std::cout << "CLASS NUMBER :" << classNumber << "IMAGE NUMBER: " << imageNumber << std::endl;
	char* trainFolder = new char[128];
#ifdef USING_WINDOWS
	if (train_test == 1) {
		//trainFolder = "C:\\Users\\rajat\\Dropbox\\WiSe18\\TDCV\\Exercise030405\\data\\task2\\train";
		//strcpy_s(trainFolder, sizeof(trainFolder), "C:\\Users\\rajat\\Dropbox\\WiSe18\\TDCV\\Exercise030405\\data\\task2\\train");
		strcpy(trainFolder, "C:\\Users\\rajat\\Dropbox\\WiSe18\\TDCV\\Exercise030405\\data\\task2\\train");
		//std::cout << "CLASS NUMBER :" << classNumber <<  "IMAGE NUMBER: " << imageNumber << std::endl;
		//std::cout << trainFolder << std::endl;
		sprintf(trainFolder, "%s\\%02d", trainFolder, classNumber);
		//std::cout << trainFolder << std::endl;
		sprintf(trainFolder, "%s\\%04d.jpg", trainFolder, imageNumber);
		//std::cout << trainFolder << std::endl;
	}
	else if(train_test == 0) {
		strcpy(trainFolder, "C:\\Users\\rajat\\Dropbox\\WiSe18\\TDCV\\Exercise030405\\data\\task2\\test");
		sprintf(trainFolder, "%s\\%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s\\%04d.jpg", trainFolder, imageNumber);
	}
	else {
		strcpy(trainFolder, "D:\\Exercise030405\\data\\task3\\train");
		sprintf(trainFolder, "%s\\%02d", trainFolder, classNumber);
		sprintf(trainFolder, "%s\\%04d.jpg", trainFolder, imageNumber);
	}
#endif
#ifdef USING_LINUX
    strcpy(trainFolder, "/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task2/train");
    sprintf(trainFolder, "%s/%02d", trainFolder, classNumber);
    sprintf(trainFolder, "%s/%04d.jpg", trainFolder, imageNumber);
#endif
	//std::cout << trainFolder << std::endl;
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
    const int cols = (int)std::max( descriptorsList[0].cols, descriptorsList[0].rows );//rows; //let's try this // HACK
    cv::Mat temp(1, cols, CV_32FC1);
    //trainingData = cv::Mat(rows, cols, CV_32FC1);
    int index = 0;
    for (std::vector<cv::Mat>::const_iterator it = descriptorsList.begin(); it != descriptorsList.end(); ++it, ++index) {
        CV_Assert( it->cols == 1 || it->rows == 1);
        if (it->cols == 1) {
            transpose(*(it), temp);
            temp.copyTo(trainingData.row(index));
        }
        else if (it->rows == 1) {
            it->copyTo(trainingData.row(index));
        }
    }
}

void decisionTree(const std::vector<cv::Mat>& descriptorsList, const std::vector<int>& labels) {
    //cv::Mat trainingData;
	const int rows = (int)descriptorsList.size();
	const int cols = (int)std::max(descriptorsList[0].cols, descriptorsList[0].rows);//rows; //let's try this // HACK
	cv::Mat trainingData = cv::Mat(rows, cols, CV_32FC1);

    alignData(descriptorsList, trainingData);
	std::cout << labels.size() << descriptorsList.size() << std::endl;
	for (int index = 0; index < labels.size(); index++) {
		std::cout << "labels are " << labels[index] << std::endl;
	}
	//try {
		static cv::Ptr<cv::ml::TrainData> trainingDataForTree = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	//} catch (cv::Exception & e) {
		//std::cerr << e.msg << std::endl; // output exception message
	//}
	cv::Ptr<cv::ml::DTrees> decisionTree = cv::ml::DTrees::create();
	decisionTree->setMaxDepth(10);
	decisionTree->setMinSampleCount(2);
	decisionTree->setRegressionAccuracy(0);
	decisionTree->setUseSurrogates(false);
	decisionTree->setMaxCategories(16);
	decisionTree->setCVFolds(0);
	decisionTree->setUse1SERule(false);
	decisionTree->setTruncatePrunedTree(false);
	decisionTree->setPriors(cv::Mat());
	decisionTree->train(trainingDataForTree);
	decisionTree->save("trainingDT.yml");
}

void randomForest(const std::vector<cv::Mat>& descriptorsList, const std::vector<int>& labels, const int& task_number) {
	int iterations = 100;
	double epsilon = 0.01f;
	const int rows = (int)descriptorsList.size();
	const int cols = (int)std::max(descriptorsList[0].cols, descriptorsList[0].rows);//rows; //let's try this // HACK
	cv::Mat trainingData = cv::Mat(rows, cols, CV_32FC1);
	alignData(descriptorsList, trainingData);
	//std::cout << labels.size() << descriptorsList.size() << std::endl;
	//for (int index = 0; index < labels.size(); index++) {
	//	std::cout << "labels are " << labels[index] << std::endl;
	//}
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
	if (task_number == 3) {
		randomTrees->save("trainedRFTask3.yml");
	}
	else {
		randomTrees->save("trainedRFTask2.yml");
	}
}

void decisionTreeTest(const std::vector <cv::Mat> &descriptorsListTest, const std::vector<int> &labels) {
	float accuracy;
	const int rows = (int)descriptorsListTest.size();
	std::cout << "number of iterations " << rows << std::endl;
	const int cols = (int)std::max(descriptorsListTest[0].cols, descriptorsListTest[0].rows);//rows; //let's try this // HACK
	cv::Mat testingData = cv::Mat(rows, cols, CV_32FC1);
	alignData(descriptorsListTest, testingData);
	cv::Ptr<cv::ml::DTrees> model = cv::ml::StatModel::load<cv::ml::DTrees>("trainingDT.yml");
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
	std::cout << "Accuracy Achieved by decisionTree Model: " << accuracy / testingData.rows << std::endl;
}

void randomForestTest(const std::vector <cv::Mat> &descriptorsListTest, const std::vector<int> &labels) {
	float accuracy;
	const int rows = (int)descriptorsListTest.size();
	std::cout << "number of iterations " << rows << std::endl;
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


int main() {
    // std::string train_folder = "/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task2/train";
    // char train_folder[128] = "/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task2/train";
    // int number_of_classes = 6;
    // int image_count = 20;
    // sprintf(train_folder, "%s/%02d", train_folder, number_of_classes-2);
    // sprintf(train_folder, "%s/%04d.jpg", train_folder, image_count);
    // std::cout << train_folder << std::endl;//
    // cv::Mat img = cv::imread("/home/rajat/Dropbox/WiSe18/TDCV/Exercise030405/data/task1/obj1000.jpg");
    // cv::Mat img2 = cv::imread(train_folder);
    // if(!img2.data) {
        // std::cout << "Could not load the image..." << std::endl;
        // return -1;
    // }
    printf("OpenCV: %s", cv::getBuildInformation().c_str());
    std::vector<int> labels;
    int numberOfClasses = 6;
    int imageNumber[] = {48, 66, 41, 52, 66, 109}; //number of images of each class #TRAINING
    std::vector<cv::Mat> imageList;
    for (int classIndex = 0; classIndex < numberOfClasses; classIndex++) {
        for (int imageIndex = 0; imageIndex <= imageNumber[classIndex]; imageIndex++) {
			//std::cout << classIndex << imageIndex << std::endl;
            loadImages(classIndex, imageIndex, imageList, TRAIN);
            labels.push_back(classIndex);
        }
        std::cout << "Number of images loaded till now: " << imageList.size() << std::endl;
        std::cout << "Number of labels assigned till now: " << labels.size() << std::endl;
    }
    std::cout << "Total number of labels assigned: " << labels.size() << std::endl;
    std::cout << "Total number of images loaded: " << imageList.size() << std::endl;
    std::vector<cv::Mat> descriptorsList;
    computeHOG(imageList, cv::Size(64, 64), descriptorsList);
    decisionTree(descriptorsList, labels);
	randomForest(descriptorsList, labels, 2); //passing 2 for task2
	int imageNumberTest[] = { 58, 76, 51, 62, 76, 119 };
	std::vector<cv::Mat> imageListTest;
	std::vector<int> labelsTest;
	for (int classIndex = 0; classIndex < numberOfClasses; classIndex++) {
		for (int imageIndex = imageNumber[classIndex] + 1; imageIndex <= imageNumberTest[classIndex]; imageIndex++) {
			loadImages(classIndex, imageIndex, imageListTest, TEST);
			labelsTest.push_back(classIndex);
		}
		std::cout << "Number of images loaded till now: " << imageListTest.size() << std::endl;
		std::cout << "Number of labels assigned till now: " << labelsTest.size() << std::endl;
	}
	std::cout << "Total number of labels assigned: " << labelsTest.size() << std::endl;
	std::cout << "Total number of images loaded: " << imageListTest.size() << std::endl;
	std::vector<cv::Mat> descriptorsListTest;
	computeHOG(imageListTest, cv::Size(64, 64), descriptorsListTest);
	decisionTreeTest(descriptorsListTest, labelsTest);
	randomForestTest(descriptorsListTest, labelsTest);
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	/*TASK3*/
	//int numberOfClassesTask3 = 4;
	//std::vector<int> labelsTask3;
	//int imageNumberTask3[] = { 52, 80, 50, 289 };
	//std::vector<cv::Mat> imageListTask3Train;
	//for (int classIndex = 0; classIndex < numberOfClassesTask3; classIndex++) {
	//	for (int imageIndex = 0; imageIndex <= imageNumberTask3[classIndex]; imageIndex++) {
	//		//std::cout << classIndex << imageIndex << std::endl;
	//		loadImages(classIndex, imageIndex, imageListTask3Train, TASK3);
	//		labelsTask3.push_back(classIndex);
	//	}
	//	std::cout << "Number of images loaded till now: " << imageListTask3Train.size() << std::endl;
	//	std::cout << "Number of labels assigned till now: " << labelsTask3.size() << std::endl;
	//}
	//std::cout << "Total number of labels assigned: " << labelsTask3.size() << std::endl;
	//std::cout << "Total number of images loaded: " << imageListTask3Train.size() << std::endl;
	//std::vector<cv::Mat> descriptorsListTask3;
	//computeHOG(imageListTask3Train, cv::Size(64, 64), descriptorsListTask3);
	//randomForest(descriptorsListTask3, labelsTask3, 3); //passing 3 for task3
	//return 0;
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