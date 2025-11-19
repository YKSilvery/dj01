#ifndef _MESSAGEMANAGER_HPP
#define _MESSAGEMANAGER_HPP
#include <glog/logging.h>
#include "SerialPort.hpp"
#include "globalParam.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <opencv2/videoio.hpp>
class MessageManager
{
private:
    Translator last_message;
    int loss_cnt;
    int message_hold_threshold = 10;
    GlobalParam *gp;
    uint32_t now_time;
    std::mutex message_lock;
    // 如果是虚拟取流，则打开视频
    cv::VideoCapture capture;
    // 获取视频总帧数，初始化当前帧数，这一步对于视频循环播放有帮助
    int totalFrames;
    int currentFrames;
    cv::VideoWriter *vw;
    uint32_t n_time;
    uint32_t start_time;
    int frame_record;
    size_t vision_size;
    size_t nav_size;

public:
    MessageManager(GlobalParam &gp);
    ~MessageManager();
    void HoldMessage(Translator &ts);
    void FakeMessage(Translator &ts);
    void read(Translator &ts, SerialPort &serialPort);
    int AvailableBytes() const;
    void readBoth(MessData &m_data, navCommand_t &n_data, SerialPort &serialPort);
    void write(Translator &ts, SerialPort &serialPort);
    void writeBoth(MessData &m_data, navInfo_t &n_data, SerialPort &serialPort);
    int CheckCrc(Translator &ts, int len);
    void UpdateCrc(Translator &ts, int len);
    void copy(Translator &src, Translator &dst);
    void initParam(int color);
    void getFrame(cv::Mat &pic, Translator translator);
    void recordFrame(cv::Mat &pic);
    void ReadLogMessage(Translator &ts, GlobalParam &gp);
    void WriteLogMessage(Translator &ts, GlobalParam &gp);
};
#endif //_MESSAGEMANAGER_HPP