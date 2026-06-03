/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Sprint 4 — Fault Injection Hook
// Added: disable_npu() implementation.

#include "congestion_aware/Topology.h"
#include "congestion_aware/Link.h"
#include <cassert>
#include <iostream>

using namespace NetworkAnalyticalCongestionAware;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);
    Link::set_event_queue(std::move(event_queue));
}

Topology::Topology() noexcept : npus_count(-1), devices_count(-1), dims_count(-1) {
    npus_count_per_dim = {};
}

int Topology::get_devices_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    return devices_count;
}

int Topology::get_npus_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    return npus_count;
}

int Topology::get_dims_count() const noexcept {
    assert(dims_count > 0);
    return dims_count;
}

std::vector<int> Topology::get_npus_count_per_dim() const noexcept {
    assert(npus_count_per_dim.size() == dims_count);
    return npus_count_per_dim;
}

std::vector<Bandwidth> Topology::get_bandwidth_per_dim() const noexcept {
    assert(bandwidth_per_dim.size() == dims_count);
    return bandwidth_per_dim;
}

void Topology::disable_npu(const DeviceId npu_id) noexcept {
    assert(0 <= npu_id && npu_id < npus_count);

    // Mark the device inoperable — Device::send() will drop all future chunks.
    devices[npu_id]->disable();

    std::cout << "[fault] NPU " << npu_id << " disabled (GSP crash injected)" << std::endl;
}

void Topology::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    const auto src = chunk->current_device()->get_id();
    assert(0 <= src && src < devices_count);

    devices[src]->send(std::move(chunk));
}

void Topology::connect(const DeviceId src,
                       const DeviceId dest,
                       const Bandwidth bandwidth,
                       const Latency latency,
                       const bool bidirectional) noexcept {
    assert(0 <= src && src < devices_count);
    assert(0 <= dest && dest < devices_count);
    assert(bandwidth > 0);
    assert(latency >= 0);

    devices[src]->connect(dest, bandwidth, latency);

    if (bidirectional) {
        devices[dest]->connect(src, bandwidth, latency);
    }
}

void Topology::instantiate_devices() noexcept {
    for (auto i = 0; i < devices_count; i++) {
        devices.push_back(std::make_shared<Device>(i));
    }
}
