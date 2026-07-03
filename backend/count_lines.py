import os

path = "storage/datasets/dataset_Diptodeep_Biswas_00d2e3e7.jsonl"
if os.path.exists(path):
    with open(path, "r") as f:
        lines = f.readlines()
    print("Number of lines in dataset_v1.jsonl:", len(lines))
    for i, line in enumerate(lines[:5]):
        print(f"Line {i+1}: {line[:200]}...")
else:
    print("dataset_v1.jsonl does not exist")
