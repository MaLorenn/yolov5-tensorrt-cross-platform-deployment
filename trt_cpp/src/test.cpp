#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // 打开默认摄像头（/dev/video0）
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    std::cout << "摄像头已打开，按 q 退出" << std::endl;

    cv::Mat frame;

    while (true)
    {
        cap >> frame;  // 读取一帧

        if (frame.empty())
        {
            std::cerr << "读取帧失败" << std::endl;
            break;
        }

        cv::imshow("Camera", frame);  // 显示画面

        if (cv::waitKey(1) == 'q')    // 按 q 退出
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}