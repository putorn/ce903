/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Sprint 4 — Fault Injection Hook
// Added: disable_npu() so the simulation can mark one NPU as inoperable.

#pragma once

#include "common/EventQueue.h"
#include "congestion_aware/Chunk.h"
#include "congestion_aware/Device.h"
#include <memory>
#include <vector>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

class Topology {
  public:
    static void set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept;

    Topology() noexcept;

    [[nodiscard]] virtual Route route(DeviceId src, DeviceId dest) const noexcept = 0;

    void send(std::unique_ptr<Chunk> chunk) noexcept;

    [[nodiscard]] int get_npus_count() const noexcept;
    [[nodiscard]] int get_devices_count() const noexcept;
    [[nodiscard]] int get_dims_count() const noexcept;
    [[nodiscard]] std::vector<int> get_npus_count_per_dim() const noexcept;
    [[nodiscard]] std::vector<Bandwidth> get_bandwidth_per_dim() const noexcept;

    // Mark npu_id as inoperable — its send() will silently drop all chunks.
    // Simulates a GSP crash: ring AllReduce stalls until the event queue drains.
    void disable_npu(DeviceId npu_id) noexcept;

  protected:
    int devices_count;
    int npus_count;
    int dims_count;
    std::vector<int> npus_count_per_dim;
    std::vector<std::shared_ptr<Device>> devices;
    std::vector<Bandwidth> bandwidth_per_dim;

    void instantiate_devices() noexcept;
    void connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency, bool bidirectional = true) noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
