//
// Created by JieYe on 3/28/23.
//
#include "cppflow/include/Model.h"
#include "cppflow/include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>

cv::Mat readImage(const std::string& path, int target_channels=3) {
    // Read the image from the path as target channels
    // current only support channels = 1 or channels = 3
    cv::Mat img;
    if(target_channels == 1) {
        img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    }
    else if(target_channels == 3) {
        cv::Mat tmp_img = cv::imread(path, cv::IMREAD_COLOR);
        // change BGR to RGB
        if (!tmp_img.empty()) {
            cv::cvtColor(tmp_img, img, cv::COLOR_BGR2RGB);
        }
    }
    else {
        std::cerr << "Error: target_channels must be 1 or 3" << std::endl;
        return img;
    }

    std::cout << "image size: " << img.size() << ", channels: " << img.channels() << std::endl;
    std::cout << "image data type: " << img.type() << std::endl;
    return img;
}

cv::Mat transformAndScaleImage(const cv::Mat& img, cv::Size target_size) {
    // Transform and scale image to range 0-1
    assert(target_size.width > 0 && target_size.height > 0);
    int scale_type = CV_32FC1;
    if(img.channels() == 3) {
        scale_type = CV_32FC3;
    }
    cv::Mat scaled_img;
    if (target_size != img.size()) {
        cv::resize(img, scaled_img, target_size);
        scaled_img.convertTo(scaled_img, scale_type, 1.f/255);
    }
    else {
        img.convertTo(scaled_img, scale_type, 1.f/255);
    }
    std::cout << "After transform and scale, image size: " << scaled_img.size() << ", channels: " << scaled_img.channels() << std::endl;
    std::cout << "After scale, image data type: " << scaled_img.type() << std::endl;
    return scaled_img;
}

cv::Mat scaleImage(const cv::Mat& img) {
    // scale image to range 0-1
    cv::Mat scaled_img;
    int scale_type = CV_32FC1;
    if(img.channels() == 3) {
        scale_type = CV_32FC3;
    }
    img.convertTo(scaled_img, scale_type, 1.f/255);
    return scaled_img;
}

// This function is used to convert an scaled image to vector<float>
std::vector<float> imageToVector_F(const cv::Mat& img) {
    // Put image in vector
    std::vector<float> img_data;
    if (img.channels() == 1) {
        img_data.assign(img.begin<float>(), img.end<float>());
    }
    else {
        img_data.reserve(img.rows * img.cols * img.channels());
        for (int i=0; i<img.rows; i++)
        {
            for (int j=0; j<img.cols; j++)
            {
                img_data.push_back(img.at<cv::Vec3f>(i, j)[0]);
                img_data.push_back(img.at<cv::Vec3f>(i, j)[1]);
                img_data.push_back(img.at<cv::Vec3f>(i, j)[2]);
            }
        }
    }
    std::cout << "After convert to vector, number of vector elements: " << img_data.size() << std::endl;
    return img_data;
}

std::vector<float> inference(const std::string& model_path, std::vector<float> input_data) {
    // Load model from the SavedModel path
    Model model(model_path);
    // Create Input Tensors and prediction Tensor
    // The tensor name is got by running saved_model_cli command
    Tensor input(model, "serving_default_input_1");
    Tensor prediction(model, "StatefulPartitionedCall");
    // Feed data to input tensor
    input.set_data(input_data);

    // Run the inference
    model.run(input, prediction);

    // return the prediction results
    return prediction.Tensor::get_data<float>();
}

// This function is used to check the prediction result
int predictClass(std::vector<float> prediction_result, float &max_prob) {
    // get the element with maximum prob
    auto max_result = std::max_element(prediction_result.begin(), prediction_result.end());
    max_prob = *max_result;
    // return the index of the element
    return std::distance(prediction_result.begin(), max_result);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_saved_model>" << " <input_image>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_image = argv[2];

    int target_channels = 3;
    int target_width = 224;
    int target_height = 224;

    // Read and preprocess image
    cv::Mat img = readImage(input_image, target_channels);
    if(img.empty()) {
        std::cerr << "Error reading image: " << input_image << std::endl;
        exit(1);
    }
    cv::Mat scaled_img = transformAndScaleImage(img, cv::Size(target_width, target_height));
    std::vector<float> img_data = imageToVector_F(scaled_img);

    auto result = inference(model_path, img_data);
    std::cout << "prediction result: [";
    for (auto i : result) {
        std::cout << i << " ";
    }
    std::cout << "]" << std::endl;

    float max_prob = 0.0f;
    int predicted_class = predictClass(result, max_prob);

    std::cout << "predicted: " << predicted_class
              << ", Probability: " << max_prob << std::endl;
}

