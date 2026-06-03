/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Sprint 4 — Fault Injection Hook
// Modified: simulation loop checks ASTRA_FAULT_NODE_ID and ASTRA_FAULT_TIME
// environment variables. When current time >= fault_time, calls
// topology->disable_npu(fault_node_id) and writes fault_events.csv to log/.
//
// Usage (via run_c_injection.py — do not call directly):
//   ASTRA_FAULT_NODE_ID=0 ASTRA_FAULT_TIME=28730170 ./AstraSim_Analytical_Congestion_Aware ...

#include "astra-sim/common/Logging.hh"
#include "common/CmdLineParser.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

int main(int argc, char* argv[]) {
    // Parse standard command line arguments
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration =
        cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    const auto logging_folder =
        cmd_line_parser.get<std::string>("logging-folder");
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");

    // Read fault injection parameters from environment variables.
    // ASTRA_FAULT_NODE_ID: which NPU to disable (-1 = no fault)
    // ASTRA_FAULT_TIME:    simulation time (ns) to inject the fault (-1 = no fault)
    const char* fault_node_env = std::getenv("ASTRA_FAULT_NODE_ID");
    const char* fault_time_env = std::getenv("ASTRA_FAULT_TIME");

    const int fault_node_id    = fault_node_env ? std::stoi(fault_node_env) : -1;
    const EventTime fault_time = fault_time_env ? (EventTime)std::stoull(fault_time_env) : 0;
    const bool fault_enabled   = (fault_node_id >= 0 && fault_time_env != nullptr);

    if (fault_enabled) {
        std::cout << "[fault] Fault injection enabled:"
                  << " NPU=" << fault_node_id
                  << " at T=" << fault_time << " ns" << std::endl;
    }

    AstraSim::LoggerFactory::init(logging_configuration, logging_folder);

    // Instantiate event queue and topology
    const auto event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);

    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);

    const auto npus_count          = topology->get_npus_count();
    const auto npus_count_per_dim  = topology->get_npus_count_per_dim();
    const auto dims_count          = topology->get_dims_count();

    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);

    auto network_apis = std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    for (int i = 0; i < npus_count; i++) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, injection_scale,
                    comm_scale, rendezvous_protocol);
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    // Fire workloads
    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
    }

    // Simulation loop — check fault injection condition at each event step.
    bool fault_injected         = false;
    EventTime fault_actual_time = 0;

    while (!event_queue->finished()) {
        event_queue->proceed();

        // Inject fault when simulation time reaches fault_time.
        // This fires mid-simulation, exactly as a GSP crash would:
        // the failed NPU's send() starts dropping chunks immediately,
        // causing any in-flight AllReduce involving it to stall.
        if (fault_enabled && !fault_injected &&
            event_queue->get_current_time() >= fault_time) {

            fault_actual_time = event_queue->get_current_time();
            topology->disable_npu(fault_node_id);
            fault_injected = true;

            std::cout << "[fault] Injected at T=" << fault_actual_time << " ns"
                      << " — NPU " << fault_node_id
                      << " will drop all future chunks." << std::endl;
        }
    }

    // Write fault_events.csv so Pod B harness (klye_code.py) can read T0.
    if (fault_injected) {
        std::ofstream fault_file(logging_folder + "/fault_events.csv");
        fault_file << "collective_instance_id,t_fault\n";
        fault_file << "gpt2," << fault_actual_time << "\n";
        std::cout << "[fault] fault_events.csv written to " << logging_folder << "/" << std::endl;
    }

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    AstraSim::LoggerFactory::shutdown();
    return 0;
}
