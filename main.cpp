#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class HAWP
{
public:
	HAWP();
	Mat detect(Mat cv_image);
private:

	void preprocess(Mat srcimg);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	const float conf_threshold = 0.95;
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Holistically-Attracted Wireframe Parsing");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

HAWP::HAWP()
{
	string model_path = "hawp_512x512_float32.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
	////ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void HAWP::preprocess(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);

	int row = dstimg.rows;
	int col = dstimg.cols;
	this->input_image_.resize(row * col * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];   ////BGR2RGB
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->mean[c]) / this->std[c];
			}
		}
	}
}

Mat HAWP::detect(Mat srcimg)
{
	this->preprocess(srcimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	// post process.																																					
	const int num_lines = this->output_node_dims[0][0];
	float *lines = ort_outputs[0].GetTensorMutableData<float>();
	float *scores = ort_outputs[1].GetTensorMutableData<float>();
	Mat dstimg = srcimg.clone();
	const int image_height = srcimg.rows;
	const int image_width = srcimg.cols;
	for (int i = 0; i < num_lines; i++)
	{
		if (scores[i] < this->conf_threshold) continue;
		int x1 = int(lines[i * 4] / 128.0*image_width);
		int y1 = int(lines[i * 4 + 1] / 128.0*image_height);
		int x2 = int(lines[i * 4 + 2] / 128.0*image_width);
		int y2 = int(lines[i * 4 + 3] / 128.0*image_height);
		line(dstimg, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
	}
	return dstimg;
}

int main()
{
	HAWP mynet;
	string imgpath = "sample.png";
	Mat srcimg = imread(imgpath);
	Mat dstimg = mynet.detect(srcimg);

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning Holistically-Attracted Wireframe Parsing in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, dstimg);
	waitKey(0);
	destroyAllWindows();
}