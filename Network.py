import ciw
import numpy 
import random
import matplotlib.pyplot as plt

NUM_SERVERS = 9
SLOW_NODES = 6
SLOW_RATE = 0.125
FAST_RATE = 0.2
TOTAL_ITERATIONS = 20
SIMULATION_TIME_SECS = 2000

# create routing table (zero matrix)
routing_table = []
for i in range(0, NUM_SERVERS):
    routing_table.append([0] * NUM_SERVERS)

# create our node distributions
arrival_dist = [ciw.dists.Exponential(rate=1)]
service_dist = [ciw.dists.Deterministic(value=0)]

# populate with other nodes
for i in range(1, NUM_SERVERS):
    r = FAST_RATE if i >= SLOW_NODES else SLOW_RATE
    arrival_dist.append(ciw.dists.NoArrivals())
    service_dist.append(ciw.dists.Exponential(rate = r))

# create table of servers
num_servers = [1] * NUM_SERVERS

def ciwNetwork(runtime, algorithm, iters, seed):
    # create the network
    N = ciw.create_network(
        arrival_distributions = arrival_dist,
        service_distributions = service_dist,
        number_of_servers = num_servers,
        routing = routing_table,
    )

    # algorithm 1
    # simple routing algorithm -> selects to node with least requests
    # the Algorithm1 class is a subclass of the ciw.Node class
    class Algorithm1(ciw.Node):
        def next_node (self, ind):
            # set least amout of requests to infinity
            min = float('inf')
            index = 2
            # iterate over requests to find the node index with least utilization
            for i in range(2, NUM_SERVERS+1):
                n = self.simulation.nodes[i]
                num = n.number_of_individuals
                if num < min:
                    min = num
                    index = i
            return self.simulation.nodes[index]

    # algorithm 2
    # faster routing algorithm -> allows faster nodes to be used first
    # the Algorithm2 class is a subclass of the ciw.Node class 
    class Aglorithm2(ciw.Node):
        def next_node(self, ind):
            # create a dictionary of the nodes with their number of requests in queue
            node_dict = {node_id: self.simulation.nodes[node_id].number_of_individuals for node_id in range(2, 10)}
            node_list = list(node_dict.items())

            # make the faster three servers come first
            for i in range(3):
                node_list.insert(0, node_list.pop())
            
            fast_servers_first = dict(node_list)

            sorted_nodes = dict(sorted(fast_servers_first.items(), key=lambda reqs: reqs[1]))

            if list(sorted_nodes.keys())[0] > SLOW_NODES:  # if a fast server is first (always return it)
                return self.simulation.nodes[list(sorted_nodes.keys())[0]]  # return this node

            else:
                for i in range(0, 10):
                    if list(sorted_nodes.keys())[i] > SLOW_NODES:  # if we have a fast server
                        if list(sorted_nodes.values())[i] - iters > list(sorted_nodes.values())[0]: 
                            return self.simulation.nodes[list(sorted_nodes.keys())[i]] # return fast server
                        else:
                            break
            return self.simulation.nodes[list(sorted_nodes.keys())[0]] # return slow server

    ciw.seed(seed)

    utilization = []

    node_class_list = [Algorithm1 if algorithm == 'Algorithm1' else Aglorithm2]
    for i in range(1, NUM_SERVERS):
        node_class_list.append(ciw.Node)

    Q = ciw.Simulation(
        N, tracker = ciw.trackers.SystemPopulation(),
        node_class = node_class_list)
    
    Q.simulate_until_max_time(runtime)
    records = Q.get_all_records()
    queue_waits = [r.waiting_time for r in records]
    completion_times = [r.service_time for r in records]

    for i in range(0, 9):
        utilization.append(Q.transitive_nodes[i].server_utilisation)

    return numpy.mean(queue_waits), numpy.mean(completion_times), utilization  # return waiting time, service time, utilization

class simulate():
    def loop(runtime, load_balancing, iteration):
        queue_list = []
        seed = []
        for i in range(0, 20):
            n = random.randint(1, 20)
            seed.append(n)
        completion_list = []
        server_util_list = []

        iters = 0
        while iters < TOTAL_ITERATIONS:
            waits, service_times, utilization = ciwNetwork(runtime, load_balancing, iteration, seed[iters])
            queue_list.append(waits)
            completion_list.append(service_times)
            server_util_list.append(utilization)
            iters += 1

        return numpy.mean(queue_list), numpy.mean(completion_list), server_util_list  # return waiting time, service time, utilization

##########
# SCRIPT #
##########

iterations = list(range(NUM_SERVERS-1))
load_balancing = ["Algorithm1", "Algorithm2"]
runtime = SIMULATION_TIME_SECS # seconds to simulate for each iteration

seed = []
for i in range(0, 20):
    seed.append(random.randint(0, 20))

# For Routing Decision 1 algorithm
req_queue_time, req_complete_time, server_utilization = simulate.loop(runtime, load_balancing[0], iterations[0])

server_util = [0] * NUM_SERVERS
x = 0

for nodes in server_utilization:
    for i in range(len(server_util)):
        server_util[i] += nodes[i]
    x += 1

# Utilization for each node
server_util[:] = [i / x for i in server_util]
    
# Average total Utilization
avg_server_util = sum(server_util[1:]) / (NUM_SERVERS - 1)

# print alg1 results
s = f'Algorithm 1:\n\tQueued Time: {req_queue_time}\n\tCompleteion Time: {req_complete_time}\n\tAverage Server Utilization: {avg_server_util}\n\tServer Utilization:\n\t\t'
for i in range(2, len(server_util)):
    s += f'Server {i}: {server_util[i]}\n\t\t'
print(s)

queue_times = []
request_completion_times = []
server_utils = []
for i in range(len(iterations)):
    queue_times.append(req_queue_time)
    request_completion_times.append(req_complete_time)
    server_utils.append(avg_server_util)

# print alg2 results
graph_queue_wait_times = []
graph_request_complete_times = []
graph_server_utils = []
for i in iterations:
    # run alg 2
    req_queue_time, req_complete_time, server_utilization = simulate.loop(runtime, load_balancing[1], i)
    graph_queue_wait_times.append(req_queue_time)
    graph_request_complete_times.append(req_complete_time)

    # create server utilization list
    server_util = [0] * 9
    x = 0

    for nodes in server_utilization:
        for j in range(len(server_util)):
            server_util[j] += nodes[j]
        x += 1

    # Utilization for each node
    for j in range(len(server_util)):
        server_util[j] /= x

    # Average Server Utilization
    avg_server_util = sum(server_util[0:8 + 1]) / 8
    graph_server_utils.append(avg_server_util)

    # print iteration results
    s = f'Algorithm 2: [Iteration {i}]\n\tQueued Time: {req_queue_time}\n\tCompleteion Time: {req_complete_time}\n\tAverage Server Utilization: {avg_server_util}\n\tServer Utilization:\n\t\t'
    for i in range(2, len(server_util)):
        s += f'Server {i}: {server_util[i]}\n\t\t'
    print(s)
    
##########
# GRAPHS #
##########

# Average Queue Wait Time
plt.scatter(iterations, queue_times,color='y')
plt.scatter(iterations, graph_queue_wait_times, color='b')
plt.plot(iterations, queue_times, color='y', label='Algorithm 1')
plt.plot(iterations, graph_queue_wait_times, color='b', label='Algorithm 2')
plt.xlabel('Iteration')
plt.ylabel('Avg Queue Time (s)')
plt.title('Comparison of Average Queue Time (s) for the Simulation')
plt.ylim(ymin=0)
plt.ylim(ymax=5.0)
plt.legend(['Algorithm 1', 'Algorithm 2'])
plt.show()

# Average Request Fulfillment Time Graph
plt.scatter(iterations, request_completion_times, color='y')
plt.scatter(iterations, graph_request_complete_times, color='b')
plt.plot(iterations, request_completion_times, color='y', label='Algorithm 1')
plt.plot(iterations, graph_request_complete_times, color='b', label='Algorithm 2')
plt.xlabel('Iteration')
plt.ylabel('Avg Request Complete Time (s)')
plt.title('Completion Time (s) for Each Iteration')
plt.ylim(ymin=0)
plt.ylim(ymax=4.0)
plt.legend(['Algorithm 1', 'Algorithm 2'])
plt.show()

# Average Server Utilization
plt.scatter(iterations, server_utils, color='y')
plt.scatter(iterations, graph_server_utils, color='b')
plt.plot(iterations, server_utils,color='y', label='Algorithm 1')
plt.plot(iterations, graph_server_utils, color='b', label='Algorithm 2')
plt.xlabel('Iteration')
plt.ylabel('Mean Total utilization')
plt.title('Avg Server Utilization in Each Iteration')
plt.ylim(ymin=0)
plt.ylim(ymax=1.0)
plt.legend(['Algorithm 1', 'Algorithm 2'])
plt.show()