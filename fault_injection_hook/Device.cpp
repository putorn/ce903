/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Sprint 4 — Fault Injection Hook
// Changed: send() checks disabled flag before forwarding chunks.
// Added: disable() and is_disabled() methods.

#include "congestion_aware/Device.h"
#include "congestion_aware/Chunk.h"
#include "congestion_aware/Link.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

Device::Device(const DeviceId id) noexcept : device_id(id), disabled(false) {
    assert(id >= 0);
}

DeviceId Device::get_id() const noexcept {
    assert(device_id >= 0);
    return device_id;
}

void Device::disable() noexcept {
    disabled = true;
}

bool Device::is_disabled() const noexcept {
    return disabled;
}

void Device::send(std::unique_ptr<Chunk> chunk) noexcept {
    // If this device has been marked inoperable (GSP crash), drop the chunk.
    // This causes any in-flight AllReduce involving this node to stall —
    // neighbouring NPUs wait forever for a chunk that never arrives.
    if (disabled) {
        return;
    }

    assert(chunk != nullptr);
    assert(chunk->current_device()->get_id() == device_id);
    assert(!chunk->arrived_dest());

    const auto next_dest    = chunk->next_device();
    const auto next_dest_id = next_dest->get_id();

    assert(connected(next_dest_id));

    links[next_dest_id]->send(std::move(chunk));
}

void Device::connect(const DeviceId id, const Bandwidth bandwidth, const Latency latency) noexcept {
    assert(id >= 0);
    assert(bandwidth > 0);
    assert(latency >= 0);
    assert(!connected(id));

    links[id] = std::make_shared<Link>(bandwidth, latency);
}

bool Device::connected(const DeviceId dest) const noexcept {
    assert(dest >= 0);
    return links.find(dest) != links.end();
}
