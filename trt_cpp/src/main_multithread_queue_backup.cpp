#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "logger.hpp"

template <typename T>
struct TRTDeleter
{
    void operator()(T* object) const
    {
        if (object != nullptr)
        {
            object->destroy();
        }
    }
};

void checkCuda(cudaError_t result, const std::string& message)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            message + "：" + cudaGetErrorString(result));
    }
}

std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("无法打开 Engine 文件：" + path);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize fileSize = file.tellg();

    if (fileSize <= 0)
    {
        throw std::runtime_error("Engine 文件为空：" + path);
    }

    file.seekg(0, std::ios::beg);

    std::vector<char> data(
        static_cast<std::size_t>(fileSize));

    if (!file.read(data.data(), fileSize))
    {
        throw std::runtime_error("读取 Engine 文件失败：" + path);
    }

    return data;
}

struct PreprocessResult
{
    std::vector<float> inputData;
    float scale;
    float padLeft;
    float padTop;
};

struct Detection
{
    cv::Rect2f box;
    float confidence;
    int classId;
};

struct CapturedFrame
{
    cv::Mat image;
    std::chrono::steady_clock::time_point captureTime;
    std::size_t frameId;
};

struct InferenceResult
{
    cv::Mat image;
    std::vector<Detection> detections;
    PreprocessResult preprocessResult;

    std::chrono::steady_clock::time_point captureTime;

    double preprocessMilliseconds;
    double inferenceMilliseconds;
    double postprocessMilliseconds;

    std::size_t frameId;
};

template <typename T>
class BoundedQueue
{
public:
    explicit BoundedQueue(std::size_t maxSize)
        : maxSize_(maxSize)
    {
    }

    bool push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        notFullCondition_.wait(
            lock,
            [this]()
            {
                return queue_.size() < maxSize_ || stopped_;
            });

        if (stopped_)
        {
            return false;
        }

        queue_.push(std::move(item));

        notEmptyCondition_.notify_one();

        return true;
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        notEmptyCondition_.wait(
            lock,
            [this]()
            {
                return !queue_.empty() || stopped_;
            });

        if (queue_.empty())
        {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();

        notFullCondition_.notify_one();

        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        stopped_ = true;

        notEmptyCondition_.notify_all();
        notFullCondition_.notify_all();
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        return queue_.size();
    }

private:
    std::size_t maxSize_;

    mutable std::mutex mutex_;
    std::condition_variable notEmptyCondition_;
    std::condition_variable notFullCondition_;

    std::queue<T> queue_;

    bool stopped_ = false;
};

PreprocessResult preprocessImage(
    const cv::Mat& image,
    int inputWidth,
    int inputHeight)
{
    if (image.empty())
    {
        throw std::runtime_error("预处理失败：输入图像为空");
    }

    const float scale = std::min(
        static_cast<float>(inputWidth) /
            static_cast<float>(image.cols),
        static_cast<float>(inputHeight) /
            static_cast<float>(image.rows));

    const int resizedWidth = static_cast<int>(
        std::round(image.cols * scale));

    const int resizedHeight = static_cast<int>(
        std::round(image.rows * scale));

    cv::Mat resizedImage;

    cv::resize(
        image,
        resizedImage,
        cv::Size(resizedWidth, resizedHeight));

    cv::Mat letterboxImage(
        inputHeight,
        inputWidth,
        CV_8UC3,
        cv::Scalar(114, 114, 114));

    const int left = (inputWidth - resizedWidth) / 2;
    const int top = (inputHeight - resizedHeight) / 2;

    resizedImage.copyTo(
        letterboxImage(
            cv::Rect(
                left,
                top,
                resizedWidth,
                resizedHeight)));

    cv::cvtColor(
        letterboxImage,
        letterboxImage,
        cv::COLOR_BGR2RGB);

    letterboxImage.convertTo(
        letterboxImage,
        CV_32FC3,
        1.0 / 255.0);

    const int channelSize = inputWidth * inputHeight;

    std::vector<float> inputData(
        3ULL * inputWidth * inputHeight);

    for (int y = 0; y < inputHeight; ++y)
    {
        for (int x = 0; x < inputWidth; ++x)
        {
            const cv::Vec3f pixel =
                letterboxImage.at<cv::Vec3f>(y, x);

            const int pixelIndex =
                y * inputWidth + x;

            inputData[0 * channelSize + pixelIndex] =
                pixel[0];

            inputData[1 * channelSize + pixelIndex] =
                pixel[1];

            inputData[2 * channelSize + pixelIndex] =
                pixel[2];
        }
    }

    return {
        std::move(inputData),
        scale,
        static_cast<float>(left),
        static_cast<float>(top)};
}

float calculateIoU(
    const cv::Rect2f& boxA,
    const cv::Rect2f& boxB)
{
    const float left =
        std::max(boxA.x, boxB.x);

    const float top =
        std::max(boxA.y, boxB.y);

    const float right =
        std::min(
            boxA.x + boxA.width,
            boxB.x + boxB.width);

    const float bottom =
        std::min(
            boxA.y + boxA.height,
            boxB.y + boxB.height);

    const float intersectionWidth =
        std::max(0.0F, right - left);

    const float intersectionHeight =
        std::max(0.0F, bottom - top);

    const float intersectionArea =
        intersectionWidth * intersectionHeight;

    const float unionArea =
        boxA.area() + boxB.area() - intersectionArea;

    if (unionArea <= 0.0F)
    {
        return 0.0F;
    }

    return intersectionArea / unionArea;
}

float calculateContainment(
    const cv::Rect2f& boxA,
    const cv::Rect2f& boxB)
{
    const float left =
        std::max(boxA.x, boxB.x);

    const float top =
        std::max(boxA.y, boxB.y);

    const float right =
        std::min(
            boxA.x + boxA.width,
            boxB.x + boxB.width);

    const float bottom =
        std::min(
            boxA.y + boxA.height,
            boxB.y + boxB.height);

    const float intersectionWidth =
        std::max(0.0F, right - left);

    const float intersectionHeight =
        std::max(0.0F, bottom - top);

    const float intersectionArea =
        intersectionWidth * intersectionHeight;

    const float smallerArea =
        std::min(boxA.area(), boxB.area());

    if (smallerArea <= 0.0F)
    {
        return 0.0F;
    }

    return intersectionArea / smallerArea;
}

std::vector<Detection> applyNMS(
    std::vector<Detection> detections,
    float iouThreshold,
    float containmentThreshold)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& first,
           const Detection& second)
        {
            return first.confidence > second.confidence;
        });

    std::vector<Detection> results;
    std::vector<bool> removed(
        detections.size(),
        false);

    for (std::size_t i = 0;
         i < detections.size();
         ++i)
    {
        if (removed[i])
        {
            continue;
        }

        results.push_back(detections[i]);

        for (std::size_t j = i + 1;
             j < detections.size();
             ++j)
        {
            if (removed[j])
            {
                continue;
            }

            if (detections[i].classId !=
                detections[j].classId)
            {
                continue;
            }

            const float iou =
                calculateIoU(
                    detections[i].box,
                    detections[j].box);

            const float containment =
                calculateContainment(
                    detections[i].box,
                    detections[j].box);

            if (iou > iouThreshold ||
                containment > containmentThreshold)
            {
                removed[j] = true;
            }
        }
    }

    return results;
}

std::vector<Detection> decodeOutput(
    const std::vector<float>& output,
    float confidenceThreshold,
    float nmsThreshold,
    float containmentThreshold)
{
    constexpr int candidateCount = 25200;
    constexpr int valuesPerCandidate = 85;
    constexpr int classCount = 80;

    std::vector<Detection> candidates;

    for (int candidateIndex = 0;
         candidateIndex < candidateCount;
         ++candidateIndex)
    {
        const float* prediction =
            output.data() +
            candidateIndex * valuesPerCandidate;

        const float centerX = prediction[0];
        const float centerY = prediction[1];
        const float width = prediction[2];
        const float height = prediction[3];
        const float objectness = prediction[4];

        int bestClassId = 0;
        float bestClassProbability = prediction[5];

        for (int classId = 1;
             classId < classCount;
             ++classId)
        {
            const float classProbability =
                prediction[5 + classId];

            if (classProbability >
                bestClassProbability)
            {
                bestClassProbability =
                    classProbability;

                bestClassId = classId;
            }
        }

        const float confidence =
            objectness * bestClassProbability;

        if (confidence < confidenceThreshold)
        {
            continue;
        }

        const float left =
            centerX - width / 2.0F;

        const float top =
            centerY - height / 2.0F;

        candidates.push_back(
            {
                cv::Rect2f(
                    left,
                    top,
                    width,
                    height),
                confidence,
                bestClassId
            });
    }

    return applyNMS(
        std::move(candidates),
        nmsThreshold,
        containmentThreshold);
}

cv::Rect restoreBoxToOriginalImage(
    const cv::Rect2f& modelBox,
    const PreprocessResult& preprocessResult,
    const cv::Size& originalSize)
{
    float x1 =
        (modelBox.x - preprocessResult.padLeft) /
        preprocessResult.scale;

    float y1 =
        (modelBox.y - preprocessResult.padTop) /
        preprocessResult.scale;

    float x2 =
        (modelBox.x + modelBox.width -
         preprocessResult.padLeft) /
        preprocessResult.scale;

    float y2 =
        (modelBox.y + modelBox.height -
         preprocessResult.padTop) /
        preprocessResult.scale;

    x1 = std::max(
        0.0F,
        std::min(
            x1,
            static_cast<float>(
                originalSize.width - 1)));

    y1 = std::max(
        0.0F,
        std::min(
            y1,
            static_cast<float>(
                originalSize.height - 1)));

    x2 = std::max(
        0.0F,
        std::min(
            x2,
            static_cast<float>(
                originalSize.width - 1)));

    y2 = std::max(
        0.0F,
        std::min(
            y2,
            static_cast<float>(
                originalSize.height - 1)));

    const int left =
        static_cast<int>(x1);

    const int top =
        static_cast<int>(y1);

    const int width =
        std::max(
            0,
            static_cast<int>(x2 - x1));

    const int height =
        std::max(
            0,
            static_cast<int>(y2 - y1));

    return cv::Rect(
        left,
        top,
        width,
        height);
}

std::string getClassName(int classId)
{
    static const std::vector<std::string> classNames =
    {
        "person", "bicycle", "car", "motorcycle",
        "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    if (classId >= 0 &&
        classId < static_cast<int>(
            classNames.size()))
    {
        return classNames[classId];
    }

    return "class_" + std::to_string(classId);
}

void drawDetections(
    cv::Mat& frame,
    const std::vector<Detection>& detections,
    const PreprocessResult& preprocessResult)
{
    for (const Detection& detection : detections)
    {
        const cv::Rect box =
            restoreBoxToOriginalImage(
                detection.box,
                preprocessResult,
                frame.size());

        if (box.width <= 0 || box.height <= 0)
        {
            continue;
        }

        const std::string label =
            getClassName(detection.classId) +
            " " +
            cv::format(
                "%.2f",
                detection.confidence);

        cv::rectangle(
            frame,
            box,
            cv::Scalar(0, 255, 0),
            2);

        int baseline = 0;

        const cv::Size labelSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
                &baseline);

        const int textTop =
            std::max(
                0,
                box.y - labelSize.height - 8);

        cv::rectangle(
            frame,
            cv::Rect(
                box.x,
                textTop,
                labelSize.width + 8,
                labelSize.height + 8),
            cv::Scalar(0, 255, 0),
            cv::FILLED);

        cv::putText(
            frame,
            label,
            cv::Point(
                box.x + 4,
                textTop + labelSize.height + 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 0, 0),
            2);
    }
}

int main(int argc, char** argv)
{
    const std::string enginePath =
        argc > 1
        ? argv[1]
        : "../../yolov5n_fp32.engine";

    constexpr int inputWidth = 640;
    constexpr int inputHeight = 640;

    constexpr std::size_t inputCount =
        1ULL * 3ULL * inputWidth * inputHeight;

    constexpr std::size_t outputCount =
        1ULL * 25200ULL * 85ULL;

    constexpr float confidenceThreshold = 0.25F;
    constexpr float nmsThreshold = 0.45F;
    constexpr float containmentThreshold = 0.80F;

    cudaStream_t stream = nullptr;
    std::vector<void*> deviceBuffers;

    try
    {
        Logger logger;

        const std::vector<char> engineData =
            readFile(enginePath);

        std::unique_ptr<
            nvinfer1::IRuntime,
            TRTDeleter<nvinfer1::IRuntime>>
            runtime(
                nvinfer1::createInferRuntime(logger));

        if (!runtime)
        {
            throw std::runtime_error(
                "创建 TensorRT Runtime 失败");
        }

        std::unique_ptr<
            nvinfer1::ICudaEngine,
            TRTDeleter<nvinfer1::ICudaEngine>>
            engine(
                runtime->deserializeCudaEngine(
                    engineData.data(),
                    engineData.size(),
                    nullptr));

        if (!engine)
        {
            throw std::runtime_error(
                "反序列化 Engine 失败");
        }

        std::unique_ptr<
            nvinfer1::IExecutionContext,
            TRTDeleter<nvinfer1::IExecutionContext>>
            context(
                engine->createExecutionContext());

        if (!context)
        {
            throw std::runtime_error(
                "创建推理上下文失败");
        }

        const int inputIndex =
            engine->getBindingIndex("images");

        const int outputIndex =
            engine->getBindingIndex("output0");

        if (inputIndex < 0 || outputIndex < 0)
        {
            throw std::runtime_error(
                "未找到 images 或 output0 绑定");
        }

        deviceBuffers.resize(
            engine->getNbBindings(),
            nullptr);

        checkCuda(
            cudaMalloc(
                &deviceBuffers[inputIndex],
                inputCount * sizeof(float)),
            "输入显存申请失败");

        checkCuda(
            cudaMalloc(
                &deviceBuffers[outputIndex],
                outputCount * sizeof(float)),
            "输出显存申请失败");

        checkCuda(
            cudaStreamCreate(&stream),
            "CUDA Stream 创建失败");

        std::vector<float> hostOutput(
            outputCount,
            0.0F);

        cv::VideoCapture camera(0, cv::CAP_V4L2);

        if (!camera.isOpened())
        {
            throw std::runtime_error(
                "无法打开 USB 摄像头 /dev/video0");
        }

        camera.set(
            cv::CAP_PROP_FRAME_WIDTH,
            640);

        camera.set(
            cv::CAP_PROP_FRAME_HEIGHT,
            480);

        camera.set(
            cv::CAP_PROP_FPS,
            30);

        std::cout << "无界面串行检测已启动" << std::endl;
        std::cout << "处理 300 帧后自动退出" << std::endl;

        double smoothedFps = 0.0;
        int processedFrames = 0;
        double totalCaptureMilliseconds = 0.0;
        double totalPreprocessMilliseconds = 0.0;
        double totalInferenceMilliseconds = 0.0;
        double totalPostprocessMilliseconds = 0.0;
        double totalDrawMilliseconds = 0.0;
        double totalFrameMilliseconds = 0.0;

        while (processedFrames < 300)
        {
            const auto frameStart =
                std::chrono::steady_clock::now();

            const auto captureStart =
                std::chrono::steady_clock::now();

            cv::Mat frame;
            camera >> frame;

            const auto captureEnd =
                std::chrono::steady_clock::now();

            if (frame.empty())
            {
                std::cerr
                    << "警告：摄像头读取帧失败"
                    << std::endl;
                continue;
            }

            const auto preprocessStart =
                std::chrono::steady_clock::now();

            PreprocessResult preprocessResult =
                preprocessImage(
                    frame,
                    inputWidth,
                    inputHeight);

            const auto preprocessEnd =
                std::chrono::steady_clock::now();

            checkCuda(
                cudaMemcpyAsync(
                    deviceBuffers[inputIndex],
                    preprocessResult.inputData.data(),
                    inputCount * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream),
                "输入复制到 GPU 失败");

            const auto inferenceStart =
                std::chrono::steady_clock::now();

            const bool inferenceSuccess =
                context->enqueueV2(
                    deviceBuffers.data(),
                    stream,
                    nullptr);

            if (!inferenceSuccess)
            {
                throw std::runtime_error(
                    "TensorRT 推理执行失败");
            }

            checkCuda(
                cudaMemcpyAsync(
                    hostOutput.data(),
                    deviceBuffers[outputIndex],
                    outputCount * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream),
                "输出复制回 CPU 失败");

            checkCuda(
                cudaStreamSynchronize(stream),
                "CUDA Stream 同步失败");

            const auto inferenceEnd =
                std::chrono::steady_clock::now();

            const auto postprocessStart =
                std::chrono::steady_clock::now();

            std::vector<Detection> detections =
                decodeOutput(
                    hostOutput,
                    confidenceThreshold,
                    nmsThreshold,
                    containmentThreshold);

            const auto postprocessEnd =
                std::chrono::steady_clock::now();

            const auto drawStart =
                std::chrono::steady_clock::now();

            drawDetections(
                frame,
                detections,
                preprocessResult);

            const auto drawEnd =
                std::chrono::steady_clock::now();

            const auto frameEnd =
                std::chrono::steady_clock::now();

            const double inferenceMilliseconds =
                std::chrono::duration<double, std::milli>(
                    inferenceEnd - inferenceStart)
                    .count();

            const double captureMilliseconds =
                std::chrono::duration<double, std::milli>(
                    captureEnd - captureStart)
                    .count();

            const double preprocessMilliseconds =
                std::chrono::duration<double, std::milli>(
                    preprocessEnd - preprocessStart)
                    .count();

            const double postprocessMilliseconds =
                std::chrono::duration<double, std::milli>(
                    postprocessEnd - postprocessStart)
                    .count();

            const double drawMilliseconds =
                std::chrono::duration<double, std::milli>(
                    drawEnd - drawStart)
                    .count();

            const double frameSeconds =
                std::chrono::duration<double>(
                    frameEnd - frameStart)
                    .count();

            const double frameMilliseconds =
                std::chrono::duration<double, std::milli>(
                    frameEnd - frameStart)
                    .count();

            const double currentFps =
                frameSeconds > 0.0
                ? 1.0 / frameSeconds
                : 0.0;

            if (smoothedFps == 0.0)
            {
                smoothedFps = currentFps;
            }
            else
            {
                smoothedFps =
                    0.90 * smoothedFps +
                    0.10 * currentFps;
            }

            totalCaptureMilliseconds += captureMilliseconds;
            totalPreprocessMilliseconds += preprocessMilliseconds;
            totalInferenceMilliseconds += inferenceMilliseconds;
            totalPostprocessMilliseconds += postprocessMilliseconds;
            totalDrawMilliseconds += drawMilliseconds;
            totalFrameMilliseconds += frameMilliseconds;

            ++processedFrames;

            if (processedFrames % 30 == 0)
            {
                std::cout
                    << "帧数：" << processedFrames
                    << "，FPS：" << std::fixed
                    << std::setprecision(2)
                    << smoothedFps
                    << "，采集：" << captureMilliseconds << " ms"
                    << "，预处理：" << preprocessMilliseconds << " ms"
                    << "，推理：" << inferenceMilliseconds << " ms"
                    << "，后处理：" << postprocessMilliseconds << " ms"
                    << "，绘框：" << drawMilliseconds << " ms"
                    << "，目标数量：" << detections.size()
                    << std::endl;
            }
        }

        if (processedFrames > 0)
        {
            const double averageCaptureMilliseconds =
                totalCaptureMilliseconds / processedFrames;

            const double averagePreprocessMilliseconds =
                totalPreprocessMilliseconds / processedFrames;

            const double averageInferenceMilliseconds =
                totalInferenceMilliseconds / processedFrames;

            const double averagePostprocessMilliseconds =
                totalPostprocessMilliseconds / processedFrames;

            const double averageDrawMilliseconds =
                totalDrawMilliseconds / processedFrames;

            const double averageFrameMilliseconds =
                totalFrameMilliseconds / processedFrames;

            const double averageFps =
                averageFrameMilliseconds > 0.0
                ? 1000.0 / averageFrameMilliseconds
                : 0.0;

            std::cout << std::endl;
            std::cout << "========== 串行性能平均结果 =========="
                      << std::endl;

            std::cout
                << std::fixed
                << std::setprecision(2)
                << "处理帧数：" << processedFrames << std::endl
                << "平均采集耗时："
                << averageCaptureMilliseconds << " ms" << std::endl
                << "平均预处理耗时："
                << averagePreprocessMilliseconds << " ms" << std::endl
                << "平均推理耗时："
                << averageInferenceMilliseconds << " ms" << std::endl
                << "平均后处理耗时："
                << averagePostprocessMilliseconds << " ms" << std::endl
                << "平均绘框耗时："
                << averageDrawMilliseconds << " ms" << std::endl
                << "平均单帧总耗时："
                << averageFrameMilliseconds << " ms" << std::endl
                << "平均端到端 FPS："
                << averageFps << std::endl;
        }

        camera.release();

        cudaStreamDestroy(stream);
        stream = nullptr;

        for (void*& buffer : deviceBuffers)
        {
            if (buffer != nullptr)
            {
                cudaFree(buffer);
                buffer = nullptr;
            }
        }
    }
    catch (const std::exception& error)
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }

        for (void*& buffer : deviceBuffers)
        {
            if (buffer != nullptr)
            {
                cudaFree(buffer);
                buffer = nullptr;
            }
        }

        std::cerr
            << "错误："
            << error.what()
            << std::endl;

        return 1;
    }

    return 0;
}