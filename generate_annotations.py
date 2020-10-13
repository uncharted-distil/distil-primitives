"""
Utility to get primitives .json files
"""
import os
import json
import importlib
import inspect

PRIMITIVES_DIR = 'distil/primitives'
OUTPUT_DIR = 'annotations'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List all the primitives
primitives = os.listdir(PRIMITIVES_DIR)
for primitive in primitives:
    f = primitive.replace('.py', '')
    try:
        lib = importlib.import_module('distil.primitives.' + f)
    except:
        continue
    for l in dir(lib):
        if 'Primitive' in l and l != 'PrimitiveBase' and l != "UnsupervisedLearnerPrimitiveBase":
            pp = getattr(lib, l)
            print(f'Extracting {l}')
            md = pp.metadata.to_json_structure()
            name = md['python_path']
            with open(os.path.join(OUTPUT_DIR, name + '.json'), 'w') as f:
                f.write(json.dumps(md, indent=4))
                f.write('\n')
