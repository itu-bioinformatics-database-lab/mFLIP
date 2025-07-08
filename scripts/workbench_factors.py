import json
import os

import requests


class WorkbenchDataPipeline:
    def __init__(self, verbose: bool = True) -> None:

        if verbose:
            print(
                f"################################ METABOLOMICS WORKBENCH PIPELINE v.1 ################################"
            )

        self.homo_sapiens_study_ids = []
        self.factors_dict = {}
        self.verbose = verbose

        self.set_homo_sapiens_study_ids()

    def set_homo_sapiens_study_ids(self) -> None:
        """
        Send a GET request to the Metabolomics Workbench API and
        retrieve a list of study IDs where the Latin name is "Homo sapiens".
        """
        url = "https://www.metabolomicsworkbench.org/rest/study/study_id/ST/species"
        response = requests.get(url)
        data = response.json()

        # Extract study IDs where Latin name == "Homo sapiens"
        self.homo_sapiens_study_ids = [
            study_data["Study ID"]
            for study_data in data.values()
            if study_data["Latin name"] == "Homo sapiens"
        ]

        if self.verbose:
            print(
                f"{len(self.homo_sapiens_study_ids)} homo sapiens study IDs are retrieved from Metabolomics Workbench API."
            )

    def get_homo_sapiens_study_factors(self) -> dict:
        """
        Send a GET request to the Metabolomics Workbench API and
        retrieve Factors for human metabolomics studies.
        """
        print(
            f"################################ STEP: FACTORS/LABEL EXTRACTION ################################"
        )

        for human_study in self.homo_sapiens_study_ids:
            try:
                url = f"https://www.metabolomicsworkbench.org/rest/study/study_id/{human_study}/factors"
                response = requests.get(url)
                response.raise_for_status()

                data = response.json()

                self.factors_dict[human_study] = {
                    v["local_sample_id"]: v["factors"] for v in data.values()
                }

                if self.verbose:
                    print(f"Extracted factors information from study {human_study}.")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from API for study {human_study}: {e}")
                continue

            except (ValueError, KeyError) as e:
                print(f"Error processing response data for study {human_study}: {e}")
                continue

            except Exception as e:
                print(f"An unexpected error occurred for study {human_study}: {e}")
                continue

        return self.factors_dict


if __name__ == "__main__":

    pipeline = WorkbenchDataPipeline(verbose=True)

    # Extract factors/labels for human metabolomic studies
    factors_dict = pipeline.get_homo_sapiens_study_factors()
    factors_dict_pth = os.path.join("workbench_human_factors.json")

    with open(factors_dict_pth, "w") as f:
        json.dump(factors_dict, f, indent=4)
