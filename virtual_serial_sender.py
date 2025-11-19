#!/usr/bin/env python3
"""
Virtual Serial Sender Script for Testing serial_bridge_node

This script simulates a lower computer sending MessData frames to serial_bridge_node.
It constructs valid MessData frames with CRC and sends them via virtual serial port.

Usage:
    python3 virtual_serial_sender.py --port /dev/pts/6 --baud 115200

Requirements:
    pip install pyserial
"""

import argparse
import struct
import serial
import time
from typing import Optional

# Constants matching serial_protocol.hpp
VISION_FRAME_HEAD = 0x71
VISION_FRAME_TAIL = 0x4C
VISION_FRAME_SIZE = 64

def crc16(buffer: bytes) -> int:
    """CRC16 calculation matching C++ implementation"""
    crc = 0
    for byte in buffer:
        crc ^= (byte << 8)
        for _ in range(8):
            tmp = crc << 1
            if crc & 0x8000:
                tmp ^= 0x1021
            crc = tmp & 0xFFFFFFFF
    return crc & 0xFFFF

def create_mess_data_frame(yaw: float = 0.5, pitch: float = 0.1,
                          status: int = 5, armor_flag: int = 1,
                          allround: bool = False, latency: float = 10.0,
                          x_c: float = 2.0, v_x: float = 0.1,
                          y_c: float = 0.5, v_y: float = 0.05,
                          z1: float = 1.0, z2: float = 1.0,
                          r1: float = 2.5, r2: float = 2.5,
                          yaw_a: float = 0.2, vyaw: float = 0.01) -> bytes:
    """
    Create a MessData frame with CRC calculation.

    Struct layout (little-endian):
    uint8_t head
    float yaw, pitch
    uint8_t status, armor_flag
    bool allround
    float latency, x_c, v_x, y_c, v_y, z1, z2, r1, r2, yaw_a, vyaw
    uint16_t crc
    uint8_t tail
    padding to 64 bytes
    """

    # Pack complete frame with CRC = 0 first
    data_with_zero_crc = struct.pack('<BffBBBfffffffffffHBBB',
                                    VISION_FRAME_HEAD,
                                    yaw, pitch,
                                    status, armor_flag,
                                    1 if allround else 0,
                                    latency, x_c, v_x, y_c, v_y, z1, z2, r1, r2, yaw_a, vyaw,
                                    0,  # CRC = 0
                                    VISION_FRAME_TAIL,
                                    0, 0)  # Padding to 61 bytes

    # Calculate CRC on first 61 bytes (including CRC field set to 0)
    crc_value = crc16(data_with_zero_crc[:61])

    # Pack with correct CRC
    data_with_crc = struct.pack('<BffBBBfffffffffffHBBB',
                               VISION_FRAME_HEAD,
                               yaw, pitch,
                               status, armor_flag,
                               1 if allround else 0,
                               latency, x_c, v_x, y_c, v_y, z1, z2, r1, r2, yaw_a, vyaw,
                               crc_value,
                               VISION_FRAME_TAIL,
                               0, 0)  # Padding

    # Pad to 64 bytes
    frame = data_with_crc.ljust(VISION_FRAME_SIZE, b'\x00')

    print("--- Sending MessData Frame ---")
    print(f"Head: {VISION_FRAME_HEAD:02X}, Tail: {VISION_FRAME_TAIL:02X}")
    print(f"CRC: {crc_value:04X}")
    print(f"Yaw: {yaw:.3f}, Pitch: {pitch:.3f}")
    print(f"Status: {status}, Armor Flag: {armor_flag}")
    print(f"Allround: {allround}, Latency: {latency:.1f}ms")
    print(f"Position: x_c={x_c:.1f}, y_c={y_c:.1f}")
    print(f"Velocities: v_x={v_x:.3f}, v_y={v_y:.3f}")
    print(f"Frame size: {len(frame)} bytes")
    # Print first 16 bytes
    print(f"First 16 bytes: {' '.join(f'{b:02X}' for b in frame[:16])}")
    # Print CRC calculation data (first 61 bytes)
    print(f"CRC data (61 bytes): {' '.join(f'{b:02X}' for b in data_with_crc[:61])}")
    print("--- End Frame ---")

    return frame

def main():
    parser = argparse.ArgumentParser(description="Virtual Serial Sender for serial_bridge_node testing")
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port to send to')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--interval', type=float, default=1.0, help='Send interval in seconds')
    parser.add_argument('--count', type=int, default=0, help='Number of frames to send (0 for infinite)')

    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Opened serial port {args.port} at {args.baud} baud")
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    frame_count = 0

    try:
        while args.count == 0 or frame_count < args.count:
            # Create frame with varying data for testing
            yaw = 0.5 + 0.1 * (frame_count % 10)
            pitch = 0.1 + 0.05 * (frame_count % 5)
            x_c = 2.0 + 0.2 * (frame_count % 8)
            y_c = 0.5 + 0.1 * (frame_count % 6)

            frame = create_mess_data_frame(
                yaw=yaw, pitch=pitch,
                x_c=x_c, y_c=y_c
            )

            # Send frame
            bytes_written = ser.write(frame)
            ser.flush()

            print(f"Sent frame #{frame_count + 1}, {bytes_written} bytes written")

            frame_count += 1

            if args.count == 0 or frame_count < args.count:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print(f"Closed serial port. Total frames sent: {frame_count}")

if __name__ == "__main__":
    main()