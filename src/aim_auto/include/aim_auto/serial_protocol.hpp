#pragma once

#include <cstdint>
#include <cstring>

namespace aim_auto
{

constexpr std::uint8_t kVisionFrameHead = 0x71;
constexpr std::uint8_t kVisionFrameTail = 0x4C;
constexpr std::uint8_t kNavFrameHead = 0x72;
constexpr std::uint8_t kNavFrameTail = 0x21;
constexpr std::uint8_t kNavInfoTail = 0x4D;

#pragma pack(push, 1)
struct MessData
{
  std::uint8_t head = kVisionFrameHead;
  float yaw = 0.0F;
  float pitch = 0.0F;
  std::uint8_t status = 0U;
  std::uint8_t armor_flag = 0U;
  bool allround = false;
  float latency = 0.0F;
  float x_c = 0.0F;
  float v_x = 0.0F;
  float y_c = 0.0F;
  float v_y = 0.0F;
  float z1 = 0.0F;
  float z2 = 0.0F;
  float r1 = 0.0F;
  float r2 = 0.0F;
  float yaw_a = 0.0F;
  float vyaw = 0.0F;
  std::uint16_t crc = 0U;
  std::uint8_t tail = kVisionFrameTail;
};

struct NavCommand
{
  NavCommand()
  {
    reserve_2 = 0U;
  }
  std::uint8_t frame_header = kNavFrameHead;
  float coo_x = 0.0F;
  float coo_y = 0.0F;
  std::uint8_t stop = 0U;
  std::uint8_t color = 0U;
  std::uint8_t eSentryState = 0U;
  std::uint8_t eSentryEvent = 0U;
  std::uint8_t constrained_defense_state = 0U;
  std::uint8_t terrian_pass = 0U;
  std::uint8_t is_revive = 0U;
  std::uint32_t reserve_2 : 24;
  std::uint32_t reserve_3 = 0U;
  std::uint32_t reserve_4 = 0U;
  std::uint32_t reserve_5 = 0U;
  std::uint32_t reserve_6 = 0U;
  std::uint32_t reserve_7 = 0U;
  std::uint32_t reserve_8 = 0U;
  std::uint32_t reserve_9 = 0U;
  std::uint32_t reserve_10 = 0U;
  std::uint32_t reserve_11 = 0U;
  std::uint32_t reserve_12 = 0U;
  std::uint32_t reserve_13 = 0U;
  std::uint8_t frame_tail = kNavFrameTail;
};

struct NavInfo
{
  NavInfo()
  {
    reserve_1 = 0U;
  }
  std::uint8_t frame_header = kNavFrameHead;
  float x_speed = 0.0F;
  float y_speed = 0.0F;
  std::uint8_t modified_flag = 0U;
  float coo_x_current = 0.0F;
  float coo_y_current = 0.0F;
  float yaw = 0.0F;
  std::uint8_t sentry_region = 0U;
  float target_x = 0.0F;
  float target_y = 0.0F;
  std::uint8_t bNoAttackEngineer = 0U;
  std::uint8_t bAttackOutpost = 0U;
  std::uint8_t bNoPatrol = 0U;
  float desire_angle = 0.0F;
  std::uint32_t reserve_1 : 8;
  std::uint32_t reserve_2 = 0U;
  std::uint32_t reserve_3 = 0U;
  std::uint32_t reserve_4 = 0U;
  std::uint32_t reserve_5 = 0U;
  std::uint32_t reserve_6 = 0U;
  std::uint32_t reserve_7 = 0U;
  std::uint8_t frame_tail = kNavInfoTail;
};
#pragma pack(pop)

constexpr std::size_t kVisionFrameSize = sizeof(MessData);

inline std::uint16_t crc16(const std::uint8_t * buffer, std::size_t length)
{
  std::uint16_t crc = 0;
  for (std::size_t j = 0; j < length; ++j) {
    crc ^= static_cast<std::uint16_t>(buffer[j]) << 8;
    for (int i = 0; i < 8; ++i) {
      const std::uint16_t tmp = crc << 1;
      crc = (crc & 0x8000) ? static_cast<std::uint16_t>(tmp ^ 0x1021) : tmp;
    }
  }
  return crc;
}

inline void updateCrc(MessData & data)
{
  data.crc = 0;
  data.crc = crc16(reinterpret_cast<const std::uint8_t *>(&data), 61);
}

inline bool verifyCrc(const MessData & data)
{
  return crc16(reinterpret_cast<const std::uint8_t *>(&data), kVisionFrameSize) == 0;
}

}  // namespace aim_auto
