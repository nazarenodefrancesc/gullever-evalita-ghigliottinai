import pandas as pd
import json

csv = pd.read_csv("/home/nazareno/Downloads/Ghigliottin-AI 2020 - gulliver.csv")

csv = csv[["Game Word 1","Game Word 2","Game Word 3","Game Word 4","Game Word 5","Game Solution"]]
csv.columns = ["w1","w2","w3","w4","w5","solution"]

json_csv = csv.to_json(orient="index")
json_loaded = json.loads(json_csv)

with open("/home/nazareno/Downloads/Ghigliottin-AI 2020 - gulliver.json", 'w') as f:
    json.dump(json_loaded, f, indent=4)

json.dumps(json_loaded, indent=4)