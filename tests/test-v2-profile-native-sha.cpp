// Device-free regression for Android's standalone-dependency-free SHA-256.
// The Phone compares this digest directly with Python hashlib on the PC.
#include "v2-profile-sha256.h"
#include <iostream>
#include <string>

static bool expect(const std::string & name,
                   const std::string & input,
                   const std::string & expected) {
    const std::string actual = v2_profile_hash::string_sha256(input);
    if (actual == expected) {
        return true;
    }
    std::cerr << "SHA256_FAIL " << name << " expected=" << expected
              << " actual=" << actual << '\n';
    return false;
}

int main() {
    bool pass = true;
    pass &= expect("empty", "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    pass &= expect("abc", "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    pass &= expect("million-a", std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    v2_profile_hash::sha256 streaming;
    streaming.update(reinterpret_cast<const uint8_t *>("a"), 1);
    streaming.update(reinterpret_cast<const uint8_t *>("bc"), 2);
    if (streaming.finish() !=
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        std::cerr << "SHA256_FAIL streaming\n";
        pass = false;
    }
    std::cout << (pass ? "V2_PROFILE_NATIVE_SHA_PASS" : "V2_PROFILE_NATIVE_SHA_FAIL") << '\n';
    return pass ? 0 : 1;
}
