import random
import numpy as np
import copy
import model_code
from evo_utils import Utils

class Mutation(object):
    def __init__(self, _log, individuals, _params=None):
        self.individuals = individuals
        self.params = _params # storing other parameters if needed, such as the index for SXB and polynomial mutation
        self.log = _log
        self.offspring = []

    def process(self, mut_offspring_num):
        offspring = self.do_mutation(num=mut_offspring_num)
        self.offspring = offspring

        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d'%(self.params['gen_no'], i)
            indi.id = indi_no

        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])

        return offspring
        
    def do_mutation(self, num):
        new_offspring_list=[]
        for _ in range(num):
            indi_pos = self._choose_one_parent()
            child = copy.deepcopy(self.individuals[indi_pos])
            new_code = model_code.mutate(child.code)
            child.code = new_code
            child.reset_acc()
            new_offspring_list.append(child)
  
        self.log.info('MUTATION finished')
        return new_offspring_list
      
    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random()*count_))
        idx2 = int(np.floor(np.random.random()*count_))
        while idx2 == idx1:
            idx2 = int(np.floor(np.random.random()*count_))
        
        if self.individuals[idx1].acc > self.individuals[idx2].acc:
            return idx1
        else:
            return idx2
        """
        idx3 = int(np.floor(np.random.random()*count_))
        while idx3 == idx1 or idx3 == idx2:
            idx3 = int(np.floor(np.random.random()*count_))

        if self.individuals[idx1].acc > self.individuals[idx2].acc:
            if self.individuals[idx1].acc > self.individuals[idx3].acc:
                return idx1
            else:
                return idx3
        else:
            if self.individuals[idx2].acc > self.individuals[idx3].acc:
                return idx2
            else:
                return idx3
        """

    def individuals_to_string(self):
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)
        