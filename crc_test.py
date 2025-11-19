#!/usr/bin/env python3
"""
CRC Test Script

Test CRC16 calculation with known data.
"""

def crc16_python(buffer: bytes) -> int:
    """Python CRC16 implementation"""
    crc = 0
    for byte in buffer:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (((crc << 1) & 0xFFFF) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def crc16_cpp_sim(buffer: bytes) -> int:
    """Simulate C++ CRC16 implementation"""
    crc = 0
    for byte in buffer:
        # crc ^= static_cast<std::uint16_t>(buffer[j]) << 8;
        crc ^= ((byte & 0xFF) << 8) & 0xFFFF
        for _ in range(8):
            # std::uint16_t tmp = crc << 1;
            tmp = (crc << 1) & 0xFFFF
            # if (crc & 0x8000) tmp ^= 0x1021;
            if crc & 0x8000:
                tmp ^= 0x1021
            # crc = tmp;
            crc = tmp & 0xFFFF
    return crc

def crc16_sender(buffer: bytes) -> int:
    """CRC16 from sender script"""
    crc = 0
    for byte in buffer:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (((crc << 1) & 0xFFFF) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

# Test data from the frame (61 bytes)
test_data = bytes.fromhex(
    "71 00 00 00 3F CD CC CC 3D 05 01 00 00 00 20 41 "
    "00 00 00 40 CD CC CC 3D 00 00 00 3F CD CC 4C 3D "
    "00 00 80 3F 00 00 80 3F 00 00 20 40 00 00 20 40 "
    "CD CC 4C 3E 0A D7 23 3C 3E 66 4C 00 00"
)

print(f"Test data length: {len(test_data)} bytes")
print(f"Test data: {' '.join(f'{b:02X}' for b in test_data[:16])}...")

sender_crc = crc16_sender(test_data)
python_crc = crc16_python(test_data)
cpp_sim_crc = crc16_cpp_sim(test_data)

print(f"Sender CRC: {sender_crc:04X}")
print(f"Python CRC: {python_crc:04X}")
print(f"C++ sim CRC: {cpp_sim_crc:04X}")
print(f"Expected from frame: 663E")
print(f"C++ calculated: DB33")