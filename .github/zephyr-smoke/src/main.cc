// Minimal heliaRT Zephyr smoke app.
//
// Exercises the public TFLM API surface enough to force the Zephyr module's
// helia_rt library to link in (otherwise `--gc-sections` would drop it and
// the smoke would degenerate into "does Zephyr build?").

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {
// Force the linker to keep a non-trivial symbol from each major
// heliaRT translation unit family.
tflite::MicroMutableOpResolver<3> g_resolver;
}  // namespace

int main(void) {
    // Touch a few op registrations so kernel TUs stay in the link graph.
    (void)g_resolver.AddFullyConnected();
    (void)g_resolver.AddSoftmax();
    (void)g_resolver.AddReshape();
    MicroPrintf("heliaRT Zephyr smoke alive: ops=%u",
                static_cast<unsigned>(g_resolver.GetRegistrationLength()));
    return 0;
}
