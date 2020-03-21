#include "BitStream.h"

class BitStream {
private:
    const uint8_t BIT_PER_BYTE = 8u;
    uint8_t* bits_container;
    uint32_t size;
    uint32_t bits_container_size;

public:
    explicit BitStream(uint32_t new_size) {
        size = new_size;
        bits_container_size = static_cast<uint32_t>(std::ceil(static_cast<double>(size) / BIT_PER_BYTE));
        bits_container = new uint8_t[bits_container_size];
    }

    void set(uint32_t index) {
        if (index >= size) {
            throw std::out_of_range(
                "BitStream operator[] got index that is larger than bits_container size"
            );
        }

        bits_container[index / BIT_PER_BYTE] |= 1u << (BIT_PER_BYTE - index % BIT_PER_BYTE - 1);
    }

    void reset(uint32_t index) {
        if (index >= size) {
            throw std::out_of_range(
                "BitStream operator[] got index that is larger than bits_container size"
            );
        }

        bits_container[index / BIT_PER_BYTE] &= ~(1u << (BIT_PER_BYTE - index % BIT_PER_BYTE));
    }

    std::string to_string() {
        std::string str;
        str.reserve(size);

        uint32_t index = 0;
        uint8_t byte;
        for (uint32_t byte_index = 0; byte_index < bits_container_size; ++byte_index) {
            byte = bits_container[byte_index];

            for (uint8_t i = 0; i < BIT_PER_BYTE; ++i) {
                if (index >= size) {
                    return str;
                }

                str.push_back(((byte & 0b10000000u) >> 7u) ? '1' : '0');
                byte <<= 1u;
                ++index;
            }
        }

        return str;
    }
};