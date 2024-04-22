import random
import time

from evo_utils import StatusUpdateTool, Utils, Log
from genetic.population import Population, Individual
from genetic.evaluate import FitnessEvaluate
from genetic.mutation import Mutation
import numpy as np
import copy

def run_evolve():
    params = {}
    params['pop_size'] = 2    # Population size
    params['max_gen'] = 3    # Maximum number of iteration generations
    evoCNN = EvolveCNN(params)
    evoCNN.do_work(params)

class EvolveCNN(object):
    def __init__(self, params):
        self.parent_pops = None
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    # type==0 means 0-th
    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()

        # for indi in self.pops.individuals:
        #     indi.acc = indi.code[1]/indi.code[0]
        # return None

        fitness.evaluate()

    def generate_offspring(self):
        cm = Mutation(Log,self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process(mut_offspring_num=self.params['pop_size'])
        
        _str = []
        for ind in offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        file_name = './populations/offspring_%02d.txt' % self.pops.gen_no
        with open(file_name, 'w') as f:
            f.write('\n'.join(_str))

        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
        pop_size = self.params['pop_size']
        elitism = 0.2
        e_count = int(pop_size * elitism)
        indi_list.sort(key=lambda x: x.acc, reverse=True)
        # descending order
        next_individuals = indi_list[0:e_count]
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for i in range(pop_size - e_count):
            idx1 = random.randrange(0, len(left_list))
            indi1 = left_list.pop(idx1)
            idx2 = random.randrange(0, len(left_list))
            indi2 = left_list.pop(idx2)
            if indi1.acc > indi2.acc:
                next_individuals.append(indi1)
            else:
                next_individuals.append(indi2)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)
        self.pops.gen_no += 1

    def do_work(self, params):
        max_gen = params['max_gen']
        Log.info('*'*25)
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population(prefix='begin', gen_no=gen_no, params=params)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1
        self.pops.gen_no += 1
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            #step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.generate_offspring()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))
            time.sleep(30)
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))

        self.params['gen_no'] = max_gen
        self.generate_offspring()

        StatusUpdateTool.end_evolution()


if __name__ == '__main__':
    run_evolve()
