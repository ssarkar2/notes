import simpy
import random
import statistics
import matplotlib.pyplot as plt



A = 0.05
B = 0.2
SIM_TIME = 10_000

random.seed(42)


'''
simpy needs generators
This is an infinite generator, as it uses "while True"
'''
def arrivals(env: simpy.Environment, queue: simpy.Store, LAMBDA):
    while True:
        r = random.expovariate(LAMBDA)
        yield env.timeout(r) # pause the simulation for "r" random units of time. other processes if any can run during that time
        arrival_timestamp = env.now # record arrival timestamp
        yield queue.put(arrival_timestamp) # FIFO queue, allowing put/get
'''
Why 2 yields?
1. pause this process till this (random) timeout occues
2. put it into a queue, wait till the queue accepts it. Note the queue is unbounded in our case so it will be accepted
Note that put returns a simpy event, and as such we are yielding it

Note: anythng that yields an event (like put in this case) must be yielded
'''



'''
for available_batches = [1,4,8]
num_items = 5 -> 8
num_items = 9 -> 8
'''
def choose_batch_size(num_items, available_batches):
    for b in available_batches:
        if b >= num_items:
            return b
    return available_batches[-1]


'''
this is a bit unrealistic
we can do something to simulate mem vs compute bound regimes, to create a more realistic model
'''
def batch_time(batch_size):
    return A * batch_size + B


def batcher(env: simpy.Environment, queue: simpy.Store, latencies, processed_count, total_waste, AVAILABLE_BATCHES):
    while True:
        # Wait for at least one item
        req = yield queue.get()
        batch = [req]

        # Drain immediately available items
        # Note that we use queue.get to trigger the first element, but the rest we get directly from queue.items
        while len(batch) < max(AVAILABLE_BATCHES) and queue.items:
            batch.append(queue.items.pop(0))

        real_items = len(batch)
        assert real_items > 0
        batch_size = choose_batch_size(real_items, AVAILABLE_BATCHES)
        assert batch_size > 0 and batch_size in AVAILABLE_BATCHES

        # Accumulate waste
        total_waste += [batch_size - real_items]

        # Process batch
        service_time = batch_time(batch_size)
        #print('model start', env.now, 'batch size', batch_size, 'service time', service_time)
        yield env.timeout(service_time)
        #print('model end', env.now)

        # Complete all requests
        for req in batch:
            latency = env.now - req
            latencies.append(latency)
            processed_count[0] += 1


def run(LAMBDA, AVAILABLE_BATCHES):
    env = simpy.Environment()
    queue = simpy.Store(env)

    latencies = []
    processed_count = [0] # Note this singleton list idiom to allow updating inside batcher
    total_waste = []

    # the 2 processes are launched here.
    # simpy lets them run concurrently. when they yield, simpy
    env.process(arrivals(env, queue, LAMBDA))
    env.process(batcher(env, queue, latencies, processed_count, total_waste, AVAILABLE_BATCHES))

    env.run(until=SIM_TIME)


    avg_latency = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    p99 = statistics.quantiles(latencies, n=100)[98]

    throughput = processed_count[0] / SIM_TIME
    waste_per_time = sum(total_waste) / SIM_TIME
    waste_per_item = sum(total_waste) / processed_count[0]

    print(f"Average latency:     {avg_latency:.3f}")
    print(f"P95 latency:         {p95:.3f}")
    print(f"P99 latency:         {p99:.3f}")
    print(f"Throughput:         {throughput:.3f} items / time")
    print(f"Waste / time:       {waste_per_time:.3f} slots / time")
    print(f"Waste per item:     {waste_per_item:.3f}")

    return avg_latency, throughput, waste_per_time, waste_per_item

if __name__ == "__main__":
    for AB in [[1,4,8], [2,4,6,8], [1,4,8,16]]:
        LAMBDAS = range(1, 20)
        avg_latencies = []
        throughputs = []
        wastes_per_time = []
        wastes_per_item = []

        for LAMBDA in LAMBDAS:
            print(f"\n=== Running simulation with LAMBDA = {LAMBDA}, AVAILABLE_BATCHES = {AB} ===")
            avg_latency, throughput, waste_per_time, waste_per_item = run(LAMBDA=LAMBDA, AVAILABLE_BATCHES=AB)
            avg_latencies.append(avg_latency)
            throughputs.append(throughput)
            wastes_per_time.append(waste_per_time)
            wastes_per_item.append(waste_per_item)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0, 0].plot(LAMBDAS, avg_latencies, marker='o')
        axs[0, 0].set_title('Average Latency vs Lambda')
        axs[0, 0].set_xlabel('Lambda')
        axs[0, 0].set_ylabel('Average Latency')

        axs[0, 1].plot(LAMBDAS, throughputs, marker='o')
        axs[0, 1].set_title('Throughput vs Lambda')
        axs[0, 1].set_xlabel('Lambda')
        axs[0, 1].set_ylabel('Throughput')

        axs[1, 0].plot(LAMBDAS, wastes_per_time, marker='o')
        axs[1, 0].set_title('Waste per Time vs Lambda')
        axs[1, 0].set_xlabel('Lambda')
        axs[1, 0].set_ylabel('Waste per Time')

        axs[1, 1].plot(LAMBDAS, wastes_per_item, marker='o')
        axs[1, 1].set_title('Waste per Item vs Lambda')
        axs[1, 1].set_xlabel('Lambda')
        axs[1, 1].set_ylabel('Waste per Item')

        fig.suptitle(f"Available batches: {AB}", fontsize=16)

        plt.tight_layout()
        plt.savefig(f"batching_simulation_{'-'.join(map(str, AB))}.png")
            


       