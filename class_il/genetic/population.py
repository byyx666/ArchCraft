import numpy as np
import hashlib
import copy

import model_code


class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no # for record the id of current individual
        self.code = []

    def initialize(self):
        init_code = model_code.init_code()
        for i in range(5):
            init_code = model_code.mutate(init_code)
        init_code = model_code.params_clip(init_code)
        self.code = init_code

    def reset_acc(self):
        self.acc = -1.0

    def __str__(self):
        flat_code = [self.code[0]]+[self.code[1]]+[j for i in self.code[2:] for j in i]
        return 'indi:'+str(self.id)+'\n'+'code:'+str(flat_code)[1:-1]+'\n'+'Acc:'+str(self.acc)

    def uuid(self):
        _str = []
        code = self.code
        flat_code = [code[0]]+[code[1]]+[j for i in code[2:] for j in i]
        for c in flat_code:
            _str.append(str(c))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_


class Population(object):
    def __init__(self, params, gen_no, pop_size=None):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        if pop_size is None:
            self.pop_size = params['pop_size']
        else:
            self.pop_size = pop_size
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


def test_individual():
    params = {}
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)


def test_population():
    params = {}
    params['pop_size'] = 20
    pop = Population(params, 0)
    pop.initialize()
    print(pop)


if __name__ == '__main__':
    #test_individual()
    test_population()


