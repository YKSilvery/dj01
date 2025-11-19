#!/usr/bin/env python3
"""
Virtual Serial Port Script for Testing serial_bridge_node

This script simulates a serial device to receive and verify data from serial_bridge_node.
It listens on a specified serial port, parses incoming MessData frames, verifies CRC,
and prints the received data for validation.

Usage:
    python3 virtual_serial.py --port /dev/ttyUSB0 --baud 115200

Requirements:
    pip install pyserial
"""

import argparse
import struct
import serial
import time
from typing import Optional

# Constants from serial_protocol.hpp
VISION_FRAME_HEAD = 0x71
VISION_FRAME_TAIL = 0x4C
VISION_FRAME_SIZE = 64  # sizeof(MessData) with padding

def crc16(buffer: bytes) -> int:
    crc = 0
    for byte in buffer:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

def parse_mess_data(data: bytes) -> Optional[dict]:
    """Parse MessData frame and verify CRC."""
    if len(data) != VISION_FRAME_SIZE:
        return None

    print(f"Parsing frame: head={data[0]:02X}, tail={data[58]:02X}, crc={data[57]:02X}{data[56]:02X}")

    # Unpack the struct (little-endian, packed)
    try:
        unpacked = struct.unpack('<BffBBB' + 'f' * 11 + 'HB', data[:59])
        head, yaw, pitch, status, armor_flag, allround, latency, x_c, v_x, y_c, v_y, z1, z2, r1, r2, yaw_a, vyaw, crc_val, tail = unpacked
        print(f"Unpacked: head={head:02X}, tail={tail:02X}, crc={crc_val:04X}")
    except struct.error as e:
        print(f"Unpack error: {e}")
        return None

    if head != VISION_FRAME_HEAD or tail != VISION_FRAME_TAIL:
        print(f"Invalid head/tail: head={head} (expected {VISION_FRAME_HEAD}), tail={tail} (expected {VISION_FRAME_TAIL})")
        return None

    # Verify CRC (set crc to 0 for calculation)
    data_for_crc = bytearray(data[:61])
    data_for_crc[56:58] = b'\x00\x00'  # Set CRC field to 0
    computed_crc = crc16(data_for_crc)
    print(f"CRC: received {crc_val:04X}, computed {computed_crc:04X}")
    if computed_crc != crc_val:
        print(f"CRC mismatch: computed {computed_crc:04X}, received {crc_val:04X}")
        return None

    return {
        'head': head,
        'yaw': yaw,
        'pitch': pitch,
        'status': status,
        'armor_flag': armor_flag,
        'allround': bool(allround),
        'latency': latency,
        'x_c': x_c,
        'v_x': v_x,
        'y_c': y_c,
        'v_y': v_y,
        'z1': z1,
        'z2': z2,
        'r1': r1,
        'r2': r2,
        'yaw_a': yaw_a,
        'vyaw': vyaw,
        'crc': crc_val,
        'tail': tail
    }

def main():
    parser = argparse.ArgumentParser(description="Virtual Serial Port for serial_bridge_node testing")
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port to use')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Opened serial port {args.port} at {args.baud} baud")
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    buffer = bytearray()
    frame_count = 0

    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"Read {len(data)} bytes from serial")
                buffer.extend(data)
                print(f"Buffer size: {len(buffer)}")

                # Look for complete frames
                while len(buffer) >= VISION_FRAME_SIZE:
                    # Find frame start
                    start_idx = -1
                    for i in range(len(buffer) - VISION_FRAME_SIZE + 1):
                        if buffer[i] == VISION_FRAME_HEAD:
                            start_idx = i
                            print(f"Found frame head at index {i}")
                            break

                    if start_idx == -1:
                        # No valid start, discard up to potential start
                        buffer = buffer[1:]
                        print("No head found, discarding byte")
                        continue

                    # Extract potential frame
                    frame_data = buffer[start_idx:start_idx + VISION_FRAME_SIZE]
                    parsed = parse_mess_data(frame_data)

                    if parsed:
                        frame_count += 1
                        print(f"\n--- Frame {frame_count} Received ---")
                        print(f"Yaw: {parsed['yaw']:.3f}, Pitch: {parsed['pitch']:.3f}")
                        print(f"Status: {parsed['status']}, Armor Flag: {parsed['armor_flag']}")
                        print(f"Allround: {parsed['allround']}, Latency: {parsed['latency']:.3f} ms")
                        print(f"Position: x_c={parsed['x_c']:.3f}, y_c={parsed['y_c']:.3f}")
                        print(f"Velocities: v_x={parsed['v_x']:.3f}, v_y={parsed['v_y']:.3f}")
                        print("--- End Frame ---")

                        # Remove processed frame
                        buffer = buffer[start_idx + VISION_FRAME_SIZE:]
                    else:
                        print("Invalid frame, skipping")
                        # Invalid frame, skip one byte
                        buffer = buffer[1:]

            time.sleep(0.01)  # Small delay to avoid busy waiting

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print(f"Closed serial port. Total frames received: {frame_count}")

if __name__ == "__main__":
    main()