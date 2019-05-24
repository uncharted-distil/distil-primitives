"""
Utility to get primitives .json files 
"""
import os
import json
import importlib

# List all the primitives
PRIMITIVES_DIR = 'distil/primitives'
primitives = os.listdir(PRIMITIVES_DIR)

def hypers(p): 
    a = 'primitive_code'
    b = 'class_type_arguments'
    c = 'Hyperparams'
    
    hp = {}
    try:
        # Get default hypers
        for h,v in primitive.metadata.query()[a][b][c].defaults().items():
            hp[h] = v
    except:
        pass
    return hp

for primitive in primitives:
    f = primitive.replace('.py', '')
    lib = importlib.import_module('distil.primitives.' + f)
    for l in dir(lib):
        if 'Primitive' in l:
            pp = getattr(lib, l)
            try:
                item = pp(hyperparams=hypers(pp))
                md = item.metadata.to_json_structure()
                name = md['python_path']
                with open('annotations/' + name + '.json', 'w') as f:
                    f.write(json.dumps(md, indent=4))
            except:
                pass