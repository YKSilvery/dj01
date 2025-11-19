#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <termios.h>

namespace aim_auto
{

class SerialPort
{
public:
  SerialPort();
  ~SerialPort();

  bool open(const std::string & device, int baud_rate, int data_bits, int stop_bits, char parity);
  void close();
  bool isOpen() const noexcept;

  std::size_t read(std::uint8_t * buffer, std::size_t length) const;
  std::size_t write(const std::uint8_t * buffer, std::size_t length) const;
  int available() const;
  int descriptor() const noexcept { return fd_; }

private:
  bool configure(int baud_rate, int data_bits, int stop_bits, char parity);
  static speed_t toSpeed(int baud_rate);

  int fd_;
  std::string device_;
};

}  // namespace aim_auto
