import random
import multiprocessing as mp
from graph_construction.create_graph_hj import get_graph

config = {}
config['config_file'] = './paths.json'
random.seed(10)

n_trial = 20
once_per_trial = 2

for _ in range(n_trial // once_per_trial):
    procs = []
    for __ in range(once_per_trial):
        config['alpha'] = random.randrange(0,100)
        config['beta'] = random.randrange(0,100)
        config['gamma'] = random.randrange(-100,0)
        p = mp.Process(target=get_graph,kwargs=config)
        p.start()
        procs.append(p)
        print(f'{once_per_trial} done')

    for p in procs:
        p.join()
        print('joined')