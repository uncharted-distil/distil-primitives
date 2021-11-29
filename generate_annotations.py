#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Utility to get primitives .json files
"""
import os
import json
import importlib

PRIMITIVES_DIR = "distil/primitives"
OUTPUT_DIR = "annotations"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List all the primitives
primitives = os.listdir(PRIMITIVES_DIR)
for primitive in primitives:
    f = primitive.replace(".py", "")
    try:
        lib = importlib.import_module("distil.primitives." + f)
    except:
        continue
    for l in dir(lib):
        if (
            "Primitive" in l
            and l != "PrimitiveBase"
            and l != "UnsupervisedLearnerPrimitiveBase"
        ):
            pp = getattr(lib, l)
            print(f"Extracting {l}")
            md = pp.metadata.to_json_structure()
            name = md["python_path"]
            with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
                f.write(json.dumps(md, indent=4))
                f.write("\n")
