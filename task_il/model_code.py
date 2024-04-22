# Methods for generating individuals

import random
import copy
import time

Max_Params_Size = 1    # Parameter constraints for generated individuals (M)

def init_code():
    code = [6, 48, [2, 4, 6, 6, 6], [2, 4, 6, 6, 6]]
    return code

def mutate(old_code):
    p_depth = 1.0/6
    p_width = 1.0/6
    p_pool = 2.0/6
    p = random.random()
    code = copy.deepcopy(old_code)
    if p < p_depth:
        dc = code[0]
        while dc == code[0]:
            change_type = random.randint(0, 1)
            r = random.random() + 1
            if change_type == 1:
                dc = code[0] * r
            else:
                dc = code[0] / r
            dc = round(dc)
        old_d = code[0]
        code[0] = dc
        if code[0] < 2:
            code[0] = 2
        for i in range(len(code[2])):
            code[2][i] = round(code[0] * code[2][i] / old_d)
        for i in range(len(code[3])):
            code[3][i] = round(code[0] * code[3][i] / old_d)
    elif p < p_depth+p_width:
        wc = code[1]
        while wc == code[1]:
            change_type = random.randint(0, 1)
            r = random.random() + 1
            if change_type == 1:
                wc = code[1] * r
            else:
                wc = code[1] / r
            wc /= 4
            wc = round(wc)
            wc *= 4
        code[1] = wc
        if code[1] < 8:
            code[1] = 8
    elif p < p_depth+p_width+p_pool:
        pool_num = 0
        for c in code[2]:
            if c != code[0]:
                pool_num += 1
        #change_type: 0-add 1-delete 2,3-change pos
        if pool_num == 0:
            change_type = 0
        elif pool_num == len(code[2]):
            change_type = random.randint(1, 3)
        else:
            change_type = random.randint(0, 3)

        if change_type == 0:
            pos = pool_num
            new_code = random.randrange(0, code[0])
        elif change_type == 1:
            pos = random.randrange(pool_num)
            new_code = code[0]
        else:
            pos = random.randrange(pool_num)
            new_code = random.randrange(0, code[0])
        code[2][pos] = new_code
        code[2].sort()
    else:
        double_num = 0
        for c in code[3]:
            if c != code[0]:
                double_num += 1
        # change_type: 0-add 1-delete 2,3-change pos
        if double_num == 0:
            change_type = 0
        elif double_num == len(code[3]):
            change_type = random.randint(1, 3)
        else:
            change_type = random.randint(0, 3)

        if change_type == 0:
            pos = double_num
            new_code = random.randrange(0, code[0])
        elif change_type == 1:
            pos = random.randrange(double_num)
            new_code = code[0]
        else:
            pos = random.randrange(double_num)
            new_code = random.randrange(0, code[0])
        code[3][pos] = new_code
        code[3].sort()
    
    try_count = 0
    while params_count(code) > Max_Params_Size:
        down_params(code)
        try_count += 1
        if try_count > 100:
            break
    return code

def params_count(code):
    pool_code = copy.deepcopy(code[2])
    double_code = copy.deepcopy(code[3])

    w = code[1]
    params = 9*3*w
    for i in range(code[0]):
        lw = w
        while i in double_code:
            double_code.remove(i)
            w *= 2
        params += 9*lw*w

    s = 32
    for pc in pool_code:
        if pc < code[0]:
          s /= 2
    params += w*s*s*100
    
    return params/1e6        

def down_params(code):
    if code[0] <=2 and code[1] <=8:
        code[3][0] = code[0]
        code[3].sort()
        return
    old_d = code[0]
    code[0] = int(code[0]*0.9)
    if code[0] < 2:
        code[0] = 2
        
    for i in range(len(code[2])):
        code[2][i] = round(code[0] * code[2][i] / old_d)
    for i in range(len(code[3])):
        code[3][i] = round(code[0] * code[3][i] / old_d)     
        
    code[1] = int(code[1]/4*0.9)*4
    if code[1] < 8:
        code[1] = 8
       

if __name__ == '__main__':

    code = init_code()
    for k in range(20):
        code1 = code
        for i in range(5):
            code1 = mutate(code1)
        print(code1)
        print(params_count(code1))
 

