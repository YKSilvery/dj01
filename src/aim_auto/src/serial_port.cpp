#include "aim_auto/serial_port.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <iostream>

namespace aim_auto
{

SerialPort::SerialPort()
: fd_(-1)
{
}

SerialPort::~SerialPort()
{
  close();
}

bool SerialPort::open(const std::string & device, int baud_rate, int data_bits, int stop_bits, char parity)
{
  close();
  device_ = device;
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd_ < 0) {
    std::cerr << "Failed to open serial device " << device << ": " << std::strerror(errno) << std::endl;
    fd_ = -1;
    return false;
  }
  ::fcntl(fd_, F_SETFL, 0);  // blocking mode
  if (!configure(baud_rate, data_bits, stop_bits, parity)) {
    close();
    return false;
  }
  return true;
}

void SerialPort::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialPort::isOpen() const noexcept
{
  return fd_ >= 0;
}

std::size_t SerialPort::read(std::uint8_t * buffer, std::size_t length) const
{
  if (fd_ < 0 || buffer == nullptr || length == 0) {
    return 0;
  }
  const ssize_t ret = ::read(fd_, buffer, length);
  return ret > 0 ? static_cast<std::size_t>(ret) : 0;
}

std::size_t SerialPort::write(const std::uint8_t * buffer, std::size_t length) const
{
  if (fd_ < 0 || buffer == nullptr || length == 0) {
    return 0;
  }
  const ssize_t ret = ::write(fd_, buffer, length);
  return ret > 0 ? static_cast<std::size_t>(ret) : 0;
}

int SerialPort::available() const
{
  if (fd_ < 0) {
    return -1;
  }
  int bytes = 0;
  return (::ioctl(fd_, FIONREAD, &bytes) == 0) ? bytes : -1;
}

speed_t SerialPort::toSpeed(int baud_rate)
{
  switch (baud_rate) {
    case 115200: return B115200;
    case 57600: return B57600;
    case 38400: return B38400;
    case 19200: return B19200;
    case 9600: return B9600;
    case 4800: return B4800;
    case 2400: return B2400;
    case 1200: return B1200;
    case 300: return B300;
    default: return B115200;
  }
}

bool SerialPort::configure(int baud_rate, int data_bits, int stop_bits, char parity)
{
  if (fd_ < 0) {
    return false;
  }

  termios options{};
  if (::tcgetattr(fd_, &options) != 0) {
    std::cerr << "tcgetattr failed for " << device_ << std::endl;
    return false;
  }

  const speed_t speed = toSpeed(baud_rate);
  ::cfsetispeed(&options, speed);
  ::cfsetospeed(&options, speed);

  options.c_cflag |= (CLOCAL | CREAD);
  options.c_cflag &= ~CSIZE;
  switch (data_bits) {
    case 6: options.c_cflag |= CS6; break;
    case 7: options.c_cflag |= CS7; break;
    case 8: default: options.c_cflag |= CS8; break;
  }

  switch (stop_bits) {
    case 2: options.c_cflag |= CSTOPB; break;
    default: options.c_cflag &= ~CSTOPB; break;
  }

  parity = static_cast<char>(std::tolower(parity));
  switch (parity) {
    case 'o':
      options.c_cflag |= (PARENB | PARODD);
      options.c_iflag |= INPCK;
      break;
    case 'e':
      options.c_cflag |= PARENB;
      options.c_cflag &= ~PARODD;
      options.c_iflag |= INPCK;
      break;
    case 'n':
    default:
      options.c_cflag &= ~PARENB;
      options.c_iflag &= ~INPCK;
      break;
  }

  options.c_iflag &= ~(BRKINT | ICRNL | ISTRIP | IXON);
  options.c_oflag &= ~OPOST;
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  options.c_cc[VTIME] = 1;  // 100ms timeout
  options.c_cc[VMIN] = 0;

  ::tcflush(fd_, TCIOFLUSH);

  if (::tcsetattr(fd_, TCSANOW, &options) != 0) {
    std::cerr << "tcsetattr failed for " << device_ << std::endl;
    return false;
  }
  return true;
}

}  // namespace aim_auto
