
import random
import csv
from statistics import mean, stdev

def tournament_selection(population, fitnesses, k=3, rng=None):
    rng = rng or random
    idxs = rng.sample(range(len(population)), k=min(k, len(population)))
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx]

def genetic_algorithm(population, fitness_fn, recombine_fn, mutate_fn,
                      ngen=200, pmut=0.1, select_fn=None, f_thres=None, seed=None):
    rng = random.Random(seed)
    if select_fn is None:
        select_fn = lambda pop, fit: tournament_selection(pop, fit, rng=rng)
    best = max(population, key=fitness_fn)
    for _ in range(ngen):
        new_pop = []
        fits = [fitness_fn(ind) for ind in population]
        while len(new_pop) < len(population):
            p1 = select_fn(population, fits)
            p2 = select_fn(population, fits)
            c1, c2 = recombine_fn(p1, p2, rng)
            if rng.random() < pmut:
                c1 = mutate_fn(c1, rng)
            if rng.random() < pmut:
                c2 = mutate_fn(c2, rng)
            new_pop.extend([c1, c2])
        population = new_pop[:len(population)]
        cand = max(population, key=fitness_fn)
        if fitness_fn(cand) > fitness_fn(best):
            best = cand
        if f_thres is not None and fitness_fn(best) >= f_thres:
            break
    return best

def create_perm_population(pop_size, n=8, rng=None):
    rng = rng or random
    pop = []
    for _ in range(pop_size):
        perm = list(range(n))
        rng.shuffle(perm)
        pop.append(perm)
    return pop

def fitness_8queens(perm):
    n = len(perm)
    total_pairs = n*(n-1)//2
    conflicts = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(perm[i]-perm[j]) == abs(i-j):
                conflicts += 1
    return total_pairs - conflicts

def pmx(a, b, rng=None):
    rng = rng or random
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    def pmx_once(p1, p2):
        child = [None]*n
        child[i:j+1] = p1[i:j+1]
        mapping = {p2[k]: p1[k] for k in range(i, j+1)}
        for idx in list(range(0, i)) + list(range(j+1, n)):
            v = p2[idx]
            while v in mapping:
                v = mapping[v]
            child[idx] = v
        return child
    return pmx_once(a, b), pmx_once(b, a)

def swap_mutation(perm, rng=None):
    rng = rng or random
    p = perm[:]
    i, j = rng.sample(range(len(p)), 2)
    p[i], p[j] = p[j], p[i]
    return p

def lecture_adjacency():
    return {
        'A': {'B','C'},
        'B': {'A','C','D'},
        'C': {'A','B'},
        'D': {'B','E'},
        'E': {'D'},
        'F': set()
    }

class MapEnv:
    def __init__(self, adj, num_colors=5):
        self.adj = adj
        self.regions = list(adj.keys())
        self.num_colors = num_colors
        edges = set()
        for a, nbrs in adj.items():
            for b in nbrs:
                edges.add(tuple(sorted((a,b))))
        self.total_edges = len(edges)

def create_color_population(pop_size, env, rng=None):
    rng = rng or random
    return [[rng.randrange(env.num_colors) for _ in env.regions] for __ in range(pop_size)]

def fitness_map_coloring(colors, env, alpha=0.2):
    color_of = {r: colors[i] for i, r in enumerate(env.regions)}
    ok = 0
    seen = set()
    for a in env.regions:
        for b in env.adj[a]:
            e = tuple(sorted((a,b)))
            if e in seen:
                continue
            seen.add(e)
            if color_of[a] != color_of[b]:
                ok += 1
    frac_ok = ok / env.total_edges if env.total_edges else 1.0
    used_colors = len(set(colors))
    return frac_ok - alpha*(used_colors/env.num_colors)

def one_point(a, b, rng=None):
    rng = rng or random
    n = len(a)
    cut = rng.randrange(1, n)
    return a[:cut]+b[cut:], b[:cut]+a[cut:]

def uniform(a, b, rng=None):
    rng = rng or random
    c1, c2 = [], []
    for i in range(len(a)):
        if rng.random() < 0.5:
            c1.append(a[i]); c2.append(b[i])
        else:
            c1.append(b[i]); c2.append(a[i])
    return c1, c2

def color_mutation(colors, env, rng=None):
    rng = rng or random
    c = colors[:]
    i = rng.randrange(len(c))
    old = c[i]
    new = rng.randrange(env.num_colors-1)
    if new >= old:
        new += 1
    c[i] = new
    return c

def run_8queens(pop_size=100, ngen=200, pmut=0.1, seed=0):
    rng = random.Random(seed)
    pop = create_perm_population(pop_size, 8, rng)
    best = genetic_algorithm(pop, fitness_8queens, pmx, swap_mutation,
                             ngen=ngen, pmut=pmut, f_thres=28, seed=seed)
    return best, fitness_8queens(best)

def run_map_coloring(pop_size=100, ngen=150, pmut=0.1, num_colors=5, uniform_cx=True, seed=0):
    rng = random.Random(seed)
    env = MapEnv(lecture_adjacency(), num_colors=num_colors)
    pop = create_color_population(pop_size, env, rng)
    cx = uniform if uniform_cx else one_point
    fit = lambda ind: fitness_map_coloring(ind, env)
    best = genetic_algorithm(pop, fit, cx, lambda c, R: color_mutation(c, env, R),
                             ngen=ngen, pmut=pmut, f_thres=1.0 - 0.2*(1/num_colors), seed=seed)
    return best, fit(best)

def grid_experiments(csv_path='ea02_ga_results.csv'):
    rows = []
    for pop in [50, 100]:
        for pmut in [0.05, 0.1, 0.2]:
            fits = []
            for r in range(100):
                _, f = run_8queens(pop_size=pop, ngen=200, pmut=pmut, seed=r)
                fits.append(f)
            rows.append(['8queens', pop, 200, pmut, round(mean(fits),4), round(stdev(fits),4)])
    for uniform_cx in [True, False]:
        for pop in [60, 120]:
            fits = []
            for r in range(100):
                _, f = run_map_coloring(pop_size=pop, ngen=150, pmut=0.1,
                                        num_colors=5, uniform_cx=uniform_cx, seed=r)
                fits.append(f)
            rows.append(['mapcolor', pop, 150, 0.1, round(mean(fits),4), round(stdev(fits),4), f'uniform={uniform_cx}'])
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem','pop_size','generations','pmut','mean_best_fitness','std_best_fitness','extra'])
        w.writerows(rows)
    return csv_path

if __name__ == '__main__':
    p = grid_experiments()
    print('CSV geschrieben:', p)
