#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

// Safe release for Windows interfaces
template<class Interface>
inline void SafeRelease(Interface*& ptr_int)
{
	if (ptr_int)
	{
		ptr_int->Release();
		ptr_int = nullptr;
	}
}

class Kinect
{
	// Pixel sizes
	static const int depth_w_ = 512;
	static const int depth_h_ = 424;
	static const int color_w_ = 1920;
	static const int color_h_ = 1080;

	// Kinect variables
	IKinectSensor* kinect_sensor;
	IDepthFrameReader* depth_frame_reader;
	IColorFrameReader* color_frame_reader;

	// Public functions
public:
	// Initializing class
	Kinect()
	{
		init_kinect();
	}
	// Release class
	~Kinect()
	{
		close_kinect();
	}

	void run()
	{
		while (true)
		{

			// Exit if ESC pressed.
			int k = waitKey(1);
			if (k == 27)
			{
				break;
			}
		}
	}

	// Private functions
private:

	void init_kinect()
	{
		HRESULT hr = GetDefaultKinectSensor(&kinect_sensor);

		// If no sensor found, throw an error
		if (FAILED(hr))
		{
			throw runtime_error("Kinect sensor could not be found!");
		}
		
		// If sensor found, initialize the different sources
		if (kinect_sensor);
		{
			init_color_source();
			init_depth_source();
		}
	}

	void close_kinect()
	{
		// Release pointers
		SafeRelease(depth_frame_reader);
		SafeRelease(color_frame_reader);

		// Close the kinect sensor
		if (kinect_sensor)
		{
			kinect_sensor->Close();
		}

		// Release kinect pointer
		SafeRelease(kinect_sensor);
	}

	void init_color_source()
	{
		// Declaring a null pointer with name "color_frame_source"
		IColorFrameSource* color_frame_source = nullptr;

		HRESULT hr = kinect_sensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = kinect_sensor->get_ColorFrameSource(&color_frame_source);
		}

		if (SUCCEEDED(hr))
		{
			hr = color_frame_source->OpenReader(&color_frame_reader);
		}

		SafeRelease(color_frame_source);
	}

	void init_depth_source()
	{
		// Declaring a null pointer with name "depth_frame_source"
		IDepthFrameSource* depth_frame_source = nullptr;

		HRESULT hr = kinect_sensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = kinect_sensor->get_DepthFrameSource(&depth_frame_source);
		}

		if (SUCCEEDED(hr))
		{
			hr = depth_frame_source->OpenReader(&depth_frame_reader);
		}

		SafeRelease(depth_frame_source);
	}

	IColorFrame* wait_for_color_frame()
	{
		while (true)
		{
			IColorFrame* color_frame = nullptr;
			HRESULT hr = color_frame_reader->AcquireLatestFrame(&color_frame);

			if (SUCCEEDED(hr))
			{
				return color_frame;
			}

			SafeRelease(color_frame);

			this_thread::sleep_for(chrono::milliseconds(1));
		}
	}

	IDepthFrame* wait_for_depth_frame()
	{
		while (true)
		{
			IDepthFrame* depth_frame = nullptr;
			HREFTYPE hr = depth_frame_reader->AcquireLatestFrame(&depth_frame);

			if (SUCCEEDED(hr))
			{
				return depth_frame;
			}

			SafeRelease(depth_frame);

			this_thread::sleep_for(chrono::milliseconds(1));
		}
	}



};

int main()
{
	Mat frame;
	VideoCapture cap;
	cap.open(0);

}