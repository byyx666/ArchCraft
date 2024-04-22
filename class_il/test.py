import json
import argparse
import os
import sys
import multiprocessing
from datetime import datetime
from trainer import train
from evo_utils import StatusUpdateTool

def run_test():
    Gpu_id = 0
    File_id = "ResAC_A-wa-C100-inc5"
    code = [10, 144, [3, 7, 8, 10, 10], [3, 8, 10, 10, 10]]    # ResAC-A, about 8.63M
    #code = [9, 12, [4, 6, 8, 8, 9], [1, 3, 5, 8, 9]]    # ResAC-B, about 0.44M
    m=TrainModel(gpu_id=Gpu_id, file_id=File_id, code=code)
    m.process()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/wa.json',
                        help='Json file of settings.')

    return parser

class TrainModel(object):
    def __init__(self, gpu_id, file_id, code):
        self.code = code
        self.file_id = file_id
        self.gpu_id = gpu_id

    def process(self):
        args = setup_parser().parse_args()
        param = load_json(args.config)
        args = vars(args)  # Converting argparse Namespace to a dict.
        args.update(param)  # Add parameters from json

        args['device'] = [str(self.gpu_id)]
        args['seed'] = [1993]

        args['depth'] = self.code[0]
        args['width'] = self.code[1]
        args['pool'] = self.code[2]
        args['double'] = self.code[3]

        aia = 0.0
        
        aia = train(args, self.file_id)
        return round(aia,3)


    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

if __name__ == '__main__':
    run_test()
