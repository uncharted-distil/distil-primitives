"""
Utility to get primitives .json files
"""
import os
import json
import importlib
import inspect

# List all the primitives
PRIMITIVES_DIR = 'distil/primitives'
primitives = os.listdir(PRIMITIVES_DIR)

for primitive in primitives:
    f = primitive.replace('.py', '')
    lib = importlib.import_module('distil.primitives.' + f)
    for l in dir(lib):
        if 'Primitive' in l and l != 'PrimitiveBase':
            pp = getattr(lib, l)
            print(f'Extracting {l}')
            md = pp.metadata.to_json_structure()
            name = md['python_path']
            with open('annotations/' + name + '.json', 'w') as f:
                f.write(json.dumps(md, indent=4))
                f.write('\n')
