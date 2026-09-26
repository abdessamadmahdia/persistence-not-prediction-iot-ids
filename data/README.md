# Datasets (not redistributed)

The three datasets are **public but not redistributed here**; their licences belong to the original providers.
Download them from the sources below and place the CSV files in this directory, or edit the path constant at
the top of each script.

## Where to get them

| Dataset | Official source | Copy used in the paper (Kaggle mirror) | File expected |
|---|---|---|---|
| TON-IoT (Train_Test Network) | UNSW Canberra — https://research.unsw.edu.au/projects/toniot-datasets | https://www.kaggle.com/datasets/fadiabuzwayed/ton-iot-train-test-network | `TON_IoT_Train_Test_Network.csv` |
| NF-BoT-IoT-v3 | University of Queensland — https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | https://www.kaggle.com/datasets/ndayisabae/nf-bot-iot-v3 | `NF-BoT-IoT-v3.csv` |
| NF-UNSW-NB15-v3 | University of Queensland — https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3 | `NF-UNSW-NB15-v3.csv` |

The experiments were run on the Kaggle copies listed above. Any copy can be used, provided it matches the row
counts below.

## Check your copy before running

| File | Rows (before de-duplication) | Attack share | Rows used after the scripts' selection and de-duplication |
|---|---|---|---|
| `TON_IoT_Train_Test_Network.csv` | 461,043 | 34.93% | 449,972 |
| `NF-BoT-IoT-v3.csv` | 16,933,808 | 99.69% | 500,141 (recent time-tail) |
| `NF-UNSW-NB15-v3.csv` | 2,365,424 | 5.40% | 494,732 (recent time-tail) |

The UQ page states 16,993,808 flows for NF-BoT-IoT-v3, but its own attack and benign counts
(16,881,819 + 51,989) sum to 16,933,808, which is the row count of the copy used here.

## Citation requirements of the providers

- **TON-IoT:** the UNSW page asks users of the TON_IoT datasets to cite the papers it lists (see the page).
- **NF-v3 datasets:** the UQ page asks users to cite the NetFlow v3 release (Luay et al.; published as
  "Time Matters: Temporal NetFlow Features for ML-Based Network Intrusion Detection", IEEE Access, 2026).

## Paths

Every script defines its dataset path in a constant near the top (e.g. `TON_PATH`, `BOT_PATH`, `UNSW_PATH`).
The defaults point at the Kaggle input directories of the mirrors above; change them to `data/<file>.csv` for
local use. No preprocessing is required before running: sorting, encoding, sanitisation, de-duplication and
sub-sampling are all performed inside the scripts.
