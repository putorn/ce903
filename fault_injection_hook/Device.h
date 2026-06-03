/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Sprint 4 — Fault Injection Hook
// Added: disabled flag so a node can be marked inoperable mid-simulation.
// When disabled=true, send() drops all chunks silently — simulating a GSP crash.

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"
#include <map>
#include <memory>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

class Device {
  public:
    explicit Device(DeviceId id) noexcept;

    [[nodiscard]] DeviceId get_id() const noexcept;

    void send(std::unique_ptr<Chunk> chunk) noexcept;

    void connect(DeviceId id, Bandwidth bandwidth, Latency latency) noexcept;

    // Mark this device as inoperable (GSP crash).
    // Once disabled, send() drops all chunks — ring AllReduce stalls.
    void disable() noexcept;

    // Check whether this device has been disabled.
    [[nodiscard]] bool is_disabled() const noexcept;

  private:
    DeviceId device_id;

    // false = healthy, true = inoperable (GSP crash injected)
    bool disabled = false;

    std::map<DeviceId, std::shared_ptr<Link>> links;

    [[nodiscard]] bool connected(DeviceId dest) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
