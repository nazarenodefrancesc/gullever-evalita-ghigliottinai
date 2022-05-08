import requests
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

	dasasets = [
		"test_datasets/Ghigliottin-AI_2020-gulliver.json",
		"test_datasets/ghigliottina_gioco_scatola.json",
		"test_datasets/ghigliottina_tv.json",
		"test_datasets/ghigliottinai_dataset.json",
	]

	rows = []
	for dataset in tqdm(dasasets):
		df = pd.read_json(dataset)

		for _, row in tqdm(df.iterrows()):
			try:
				w1 = row["w1"]
				w2 = row["w2"]
				w3 = row["w3"]
				w4 = row["w4"]
				w5 = row["w5"]
				solution = row["solution"]
				url = f"http://93.51.19.23:2323/solve_guillotine?guillotine={w1},{w2},{w3},{w4},{w5}"
				response = requests.get(url)
				solutions = response.json()["first_10_solutions"]
				candidate_solutions = [sol[0] for sol in solutions]
				solution_present = solution in candidate_solutions
				record = {
					"w1": w1,
					"w2": w2,
					"w3": w3,
					"w4": w4,
					"w5": w5,
					"solution_present": solution_present,
					"solution": solution,
					"candidate_solutions": candidate_solutions,
				}
				rows.append(record)
			except Exception as e:
				print(e)

	output_df = pd.DataFrame(rows)
	output_df.to_csv("training_set.csv", index=False)

