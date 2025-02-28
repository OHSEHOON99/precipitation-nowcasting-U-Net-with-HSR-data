import numpy as np
import os
import logging
import pickle as pkl

from numba import jit

# Adding the parent directory to the system path so that we can import modules from there
import sys
sys.path.append('../')

from utils import *


@jit(nopython=True)
def _get_hit_miss_counts_numba(prediction, truth, thresholds):
    """
    Calculate true positive (TP), false positive (FP), true negative (TN), and false negative (FN) counts for each
    threshold in a binary classification problem using Numba for performance optimization.

    Parameters:
    - prediction (np.ndarray): Model predictions, shape (batch_size, seqlen, height, width).
    - truth (np.ndarray): Ground truth labels, shape (batch_size, seqlen, height, width).
    - thresholds (list): List of threshold values for binary classification.

    Returns:
    - ret (np.ndarray): Counts for each threshold, shape (seqlen, batch_size, threshold_num, 4).
    """
    batch_size, seqlen, height, width = prediction.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seqlen, batch_size, threshold_num, 4), dtype=np.int32)

    for j in range(batch_size):
        for i in range(seqlen):
            for m in range(height):
                for n in range(width):
                    for k in range(threshold_num):
                        bpred = prediction[j][i][m][n] >= thresholds[k]
                        btruth = truth[j][i][m][n] >= thresholds[k]
                        ind = (1 - btruth) * 2 + (1 - bpred)
                        ret[i][j][k][ind] += 1
                        # Use indexing to update counters based on prediction and truth
                        # When btrue == True and bpred == True (TP): index is 0
                        # When btrue == True and bpred == False (FN): index is 1
                        # When btrue == False and bpred == True (FP): index is 2
                        # When btrue == False and bpred == False (TN): index is 3
    TP = ret[:, :, :, 0]
    FN = ret[:, :, :, 1]
    FP = ret[:, :, :, 2]
    TN = ret[:, :, :, 3]

    return TP, FN, FP, TN


def get_hit_miss_counts_numba(prediction, truth, thresholds):
    """This function calculates the overall TP and TN for the prediction, which could be used
    to get the skill scores and threat scores:

    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters:
    - prediction (np.ndarray): Model predictions, shape (batch_size, seq_len, height, width).
    - truth (np.ndarray): Ground truth labels, shape (batch_size, seq_len, height, width).
    - thresholds (list or tuple): List of threshold values for binary classification.

    Returns:
    - TP (np.ndarray): True positive counts, shape (seq_len, batch_size, len(thresholds)).
    - TN (np.ndarray): True negative counts, shape (seq_len, batch_size, len(thresholds)).
    - FP (np.ndarray): False positive counts, shape (seq_len, batch_size, len(thresholds)).
    - FN (np.ndarray): False negative counts, shape (seq_len, batch_size, len(thresholds)).
    """
    thresholds = [dBZ_to_pixel(rfrate_to_dBZ(ele)) for ele in thresholds]
    thresholds = sorted(thresholds)
    TP, FN, FP, TN = _get_hit_miss_counts_numba(prediction=prediction, truth=truth, thresholds=thresholds)

    return TP, FN, FP, TN


class Evaluater:
    def __init__(self, seq_len):
        """
        Initializes an instance of the Evaluater class.

        Parameters:
        - seq_len (int): Sequence length.
        """
        self.threshold = [0.5, 2, 5, 10, 30]
        self.seq_len = seq_len
        self.begin()


    def begin(self):
        """
        Initializes evaluation metrics.
        """
        self.total_TP = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # TP: True Positives
        self.total_FN = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # FN: False Negatives
        self.total_FP = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # FP: False Positives
        self.total_TN = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # TN: True Negatives

        self.mse = np.zeros((self.seq_len,), dtype=np.float32)  # Mean Squared Error (MSE)
        self.mae = np.zeros((self.seq_len,), dtype=np.float32)  # Mean Absolute Error (MAE)
        self.total_batch_num = 0  # Total number of sequences


    def clear_all(self):
        """
        Resets all evaluation metrics.
        """
        self.total_TP[:] = 0
        self.total_FN[:] = 0
        self.total_FP[:] = 0
        self.total_TN[:] = 0
        self.mse[:] = 0
        self.mae[:] = 0
        self.total_batch_num = 0


    def update(self, gt, pred):
        """
        Updates evaluation metrics with new ground truth and prediction.

        Parameters:
        - gt (np.ndarray): Ground truth, shape (batch_size, seq_len, height, width).
        - pred (np.ndarray): Model predictions, shape (batch_size, seq_len, height, width).
        """
        batch_size = gt.shape[0]
        self.total_batch_num += batch_size

        # Compute MSE and MAE for each time step (axis 0: batch, axis 2: height, axis 3: width)
        mse = np.square(pred - gt).mean(axis=(0, 2, 3))  # Averaged over (seq_len,)
        mae = np.abs(pred - gt).mean(axis=(0, 2, 3))  # Averaged over (seq_len,)

        # Accumulate MSE and MAE
        self.mse += mse
        self.mae += mae

        # Compute True Positives, False Negatives, False Positives, True Negatives
        TP, FN, FP, TN = get_hit_miss_counts_numba(prediction=pred, truth=gt, thresholds=self.threshold)

        # Update TP, TN, FP, FN for each category
        self.total_TP += TP.sum(axis=1)
        self.total_TN += TN.sum(axis=1)
        self.total_FP += FP.sum(axis=1)
        self.total_FN += FN.sum(axis=1)


    def calculate_stat(self):
        """
        Calculates various evaluation metrics such as precision, FAR, CSI, HSS, GSS, MSE, and MAE.

        Returns:
        Tuple containing:
        - precision (np.ndarray): Precision scores.
        - recall (np.ndarray): Recall scores.
        - f1 (np.ndarray): F1 scores.
        - far (np.ndarray): False Alarm Rates.
        - csi (np.ndarray): Critical Success Index scores.
        - hss (np.ndarray): Heidke Skill Score.
        - gss (np.ndarray): Gilbert Skill Score.
        - mse (np.ndarray): Mean Squared Error.
        - mae (np.ndarray): Mean Absolute Error.
        """
        TP = self.total_TP.astype(np.float64)
        TN = self.total_TN.astype(np.float64)
        FP = self.total_FP.astype(np.float64)
        FN = self.total_FN.astype(np.float64)

        epsilon = 1e-7  # A small value to prevent division by zero
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        far = FP / (TP + FP)
        csi = TP / (TP + FP + FN)
        n = TP + FP + FN + TN
        aref = (TP + FP) * (TP + FN) / n
        gss = (TP - aref) / (TP + FP + FN - aref)
        hss = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
        mse = self.mse / self.total_batch_num
        mae = self.mae / self.total_batch_num

        return precision, recall, f1, far, csi, hss, gss, mse, mae


    def print_stat_readable(self):
        """
        Prints human-readable evaluation metrics in a table-like format with fixed-width columns.
        """
        logging.getLogger().setLevel(logging.INFO)

        # Calculate stats
        precision, recall, f1, far, csi, hss, gss, mse, mae = self.calculate_stat()

        # Define headers
        headers = ["Metric"] + [f">{threshold:.1f}" for threshold in self.threshold]

        # Define fixed column widths
        metric_col_width = 15  # Width for the Metric column
        col_width = 12         # Width for the other columns

        # Collect the rows for each metric
        rows = [
            ["TP Count"]   + [f"{self.total_TP[:, i].sum():>{col_width}.0f}" for i in range(len(self.threshold))],
            ["Precision"]  + [f"{precision[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["Recall"]     + [f"{recall[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["F1 Score"]   + [f"{f1[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["FAR"]        + [f"{far[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["CSI"]        + [f"{csi[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["GSS"]        + [f"{gss[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))],
            ["HSS"]        + [f"{hss[:, i].mean():>{col_width}.4f}" for i in range(len(self.threshold))]
        ]

        # Print table header
        header_str = f"{'Metric':>{metric_col_width}}" + " | " + " | ".join([f"{header:>{col_width}}" for header in headers[1:]])
        logging.info(header_str)
        
        # Print separator
        separator_length = metric_col_width + (len(self.threshold) * (col_width + 3)) - 1
        logging.info("-" * separator_length)

        # Print each row
        for row in rows:
            row_str = f"{row[0]:>{metric_col_width}}" + " | " + " | ".join(row[1:])
            logging.info(row_str)

        # Print another separator before MSE and MAE
        logging.info("-" * separator_length)

        # Print MSE and MAE as single values without repeating across thresholds
        mse_str = f"{'MSE':>{metric_col_width}} | {mse.mean():>{col_width}.4f}"
        mae_str = f"{'MAE':>{metric_col_width}} | {mae.mean():>{col_width}.4f}"

        logging.info(mse_str)
        logging.info(mae_str)

        return precision, recall, f1, far, csi, hss, gss, mse, mae

    def save_pkl(self, path):
        """
        Saves the Evaluater object to a pickle file.

        Parameters:
        - path (str): File path for saving the pickle file.
        """
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info(f"Saving Evaluater to {path}")
        pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()


    def save_txt_readable(self, path):
        """
        Saves human-readable evaluation metrics to a text file.

        Parameters:
        - path (str): File path for saving the text file.
        """
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        precision, far, csi, hss, gss, mse, mae = self.calculate_stat()
        f = open(path, 'w')
        logging.info(f"Saving readable txt of Evaluater to {path}")
        f.write(f"Total Sequence Num: {self.total_batch_num}, Seq Len: {self.seq_len}\n")
        for (i, threshold) in enumerate(self.threshold):
            f.write(f"Threshold = {threshold}:\n")
            f.write(f"   Precision: {list(precision[:, i])}\n")
            f.write(f"   FAR: {list(far[:, i])}\n")
            f.write(f"   CSI: {list(csi[:, i])}\n")
            f.write(f"   GSS: {list(gss[:, i])}\n")
            f.write(f"   HSS: {list(hss[:, i])}\n")
            f.write(f"   Precision stat: avg {precision[:, i].mean()}/final {precision[-1, i]}\n")
            f.write(f"   FAR stat: avg {far[:, i].mean()}/final {far[-1, i]}\n")
            f.write(f"   CSI stat: avg {csi[:, i].mean()}/final {csi[-1, i]}\n")
            f.write(f"   GSS stat: avg {gss[:, i].mean()}/final {gss[-1, i]}\n")
            f.write(f"   HSS stat: avg {hss[:, i].mean()}/final {hss[-1, i]}\n")
        f.write(f"MSE: {list(mse)}\n")
        f.write(f"MAE: {list(mae)}\n")
        f.write(f"MSE stat: avg {mse.mean()}/final {mse[-1]}\n")
        f.write(f"MAE stat: avg {mae.mean()}/final {mae[-1]}\n")
        f.close()


    def save(self, prefix):
        """
        Saves the Evaluater object to both a readable text file and a pickle file.

        Parameters:
        - prefix (str): Prefix for file names.
        """
        self.save_txt_readable(prefix + ".txt")
        self.save_pkl(prefix + ".pkl")