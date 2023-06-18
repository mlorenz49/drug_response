import pandas as pd
import numpy as np


class Dataprep:

    def __init__(self, drug_annotation_path, mean_logIC50_path, tpm_path):
        self.drugs = pd.read_csv(drug_annotation_path)
        self.logIC50 = pd.read_csv(mean_logIC50_path)
        self.tpm = pd.read_csv(tpm_path, sep=",", encoding="utf-8")

    def convert_data(self):
        tpm = self.tpm
        tpm.drop(["tx", "gene_id"], inplace=True, axis=1)
        tpm_T = tpm.T
        tpm_T.sort_index(ascending=True, inplace=True)

        logIC50 = self.logIC50

        logIC50.sort_values("CELL_NAME", ascending=True, inplace=True)

        logIC50.set_index(["CELL_NAME"], inplace=True)

        return tpm_T, logIC50

    def split_data(self, x_data, y_data, split=[60, 20, 20]):
        train_nr = int(round(len(x_data) / 100 * split[0], 0))
        val_nr = int(round(len(x_data) / 100 * split[1], 0))

        x_data_train = x_data[:train_nr, :]
        x_data_test = x_data[train_nr + 1:train_nr + val_nr + 1, :]

        y_data_train = y_data[:train_nr, :]
        y_data_test = y_data[train_nr + 1:train_nr + val_nr + 1, :]

        return x_data_train, x_data_test, y_data_train, y_data_test

    def extract_cell_lines(self, tpm_Tx, logIC50_drugy):
        cell_lines = list(logIC50_drugy.index.values)
        tpm_T_cell_line = tpm_Tx[tpm_Tx.index.isin(cell_lines)]
        tpm_T_cell_line = tpm_T_cell_line.to_numpy()

        return tpm_T_cell_line


if __name__ == "__main__":
    data_preper = Dataprep("well_annotated_drugs.csv",
                           "nci60_logIC50s_small.csv",
                           "nci60_tpm.tsv")

    # data_preper.drugs["NSC"]

    tpm_T, logIC50 = data_preper.convert_data()
    logIC50_drug = logIC50[logIC50["NSC"] == 12198]

    # extract cell lines tested with drug
    tpm_T_np = data_preper.extract_cell_lines(tpm_T, logIC50_drug)

    logIC50_drug.drop(["NSC"], inplace=True, axis=1)
    logIC50_drug_np = logIC50_drug.to_numpy()
