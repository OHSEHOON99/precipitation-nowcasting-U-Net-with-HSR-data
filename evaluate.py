import numpy as np
import logging
import torch
import pickle as pkl

from numba import jit
from tqdm import tqdm
from utils import *


@torch.inference_mode()
def evaluate(net, dataloader, device, criterion, amp):
    """
    Evaluate the model on the validation set.

    Parameters:
    - net (torch.nn.Module): The neural network model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - device (torch.device): Device on which to perform the evaluation.
    - criterion: Loss function used for evaluation.
    - amp (bool): Whether to use Automatic Mixed Precision (AMP) for evaluation.

    Returns:
    - val_loss (float): Average validation loss.
    """
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, label = batch['input'], batch['label']

            # move image and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device=device, dtype=torch.float32)

            # predict the mask
            label_pred = net(image).float()
            loss = criterion(label_pred, label)
            val_loss += loss.item()

    net.train()

    return val_loss / max(num_val_batches, 1)


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
                        # The above code is the same as:
                        # TP
                        # ret[i][j][k][0] += bpred * btruth
                        # FP
                        # ret[i][j][k][1] += (1 - bpred) * btruth
                        # TN
                        # ret[i][j][k][2] += bpred * (1 - btruth)
                        # FN
                        # ret[i][j][k][3] += (1 - bpred) * (1- btruth)
    return ret


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
    thresholds = [dBZ_to_grayscale(rfrate_to_dBZ(ele))/255.0 for ele in thresholds] 
    thresholds = sorted(thresholds)
    ret = _get_hit_miss_counts_numba(prediction=prediction, truth=truth, thresholds=thresholds)

    return ret[:, :, :, 0], ret[:, :, :, 1], ret[:, :, :, 2], ret[:, :, :, 3]


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
        self.total_TP = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # TP : True Positives
        self.total_FN = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # FN : False Negatives
        self.total_FP = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # FP : False Positives
        self.total_TN = np.zeros((self.seq_len, len(self.threshold)), dtype=int)  # TN : True Negatives

        self.mse = np.zeros((self.seq_len,), dtype=np.float32)  # MSE
        self.mae = np.zeros((self.seq_len,), dtype=np.float32)  # MAE
        self.total_batch_num = 0  # Total number of sequences


    def clear_all(self):
        """
        Clears all evaluation metrics.
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

        mse = np.square(pred - gt).sum(axis=(0, 2, 3))
        mae = np.abs(pred - gt).sum(axis=(0, 2, 3))

        self.mse += mse
        self.mae += mae

        TP, TN, FP, FN = get_hit_miss_counts_numba(prediction=pred, truth=gt, thresholds=self.threshold)

        self.total_TP += TP.sum(axis=(1))
        self.total_FN += FN.sum(axis=(1))
        self.total_FP += FP.sum(axis=(1))
        self.total_TN += TN.sum(axis=(1))


    def calculate_f1_score(self):
        """
        Calculates precision, recall, and F1 score for each threshold.

        Returns:
        - precision (np.ndarray): Precision scores.
        - recall (np.ndarray): Recall scores.
        - f1 (np.ndarray): F1 scores.
        """
        a = self.total_TP.astype(np.float64)
        c = self.total_FN.astype(np.float64)
        d = self.total_TN.astype(np.float64)

        precision = a / (a + c)
        recall = a / (a + d)
        f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1


    def calculate_stat(self):
        """
        Calculates various evaluation metrics such as precision, FAR, CSI, HSS, GSS, MSE, and MAE.

        Returns:
        Tuple containing:
        - precision (np.ndarray): Precision scores.
        - recall (np.ndarray): Recall scores.
        - f1 (np.ndarray): F1 scores.
        - far (np.ndarray): False alarm rates.
        - csi (np.ndarray): Critical Success Index scores.
        - hss (np.ndarray): Heidke Skill Score.
        - gss (np.ndarray): Gilbert Skill Score.
        - mse (np.ndarray): Mean Squared Error.
        - mae (np.ndarray): Mean Absolute Error.
        """
        a = self.total_TP.astype(np.float64)
        b = self.total_FP.astype(np.float64)
        c = self.total_FN.astype(np.float64)
        d = self.total_TN.astype(np.float64)

        precision = a / (a + c)
        recall = a / (a + d)
        f1 = (2 * precision * recall) / (precision + recall)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        mse = self.mse / self.total_batch_num
        mae = self.mae / self.total_batch_num

        return precision, recall, f1, far, csi, hss, gss, mse, mae


    def print_stat_readable(self, prefix=""):
        """
        Prints human-readable evaluation metrics.

        Parameters:
        - prefix (str): Prefix for log messages.
        """
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Total Sequence Number: {self.total_batch_num}")

        precision, recall, f1, far, csi, hss, gss, mse, mae = self.calculate_stat()

        logging.info("   TP: " + ', '.join([">%g:%g/%g" % (threshold,
                                                             self.total_TP[:, i].mean(),
                                                             self.total_TP[-1, i])
                                             for i, threshold in enumerate(self.threshold)]))
        logging.info("   Precision: " + ', '.join([">%g:%g/%g" % (threshold, precision[:, i].mean(), precision[-1, i])
                                  for i, threshold in enumerate(self.threshold)]))
        logging.info("   Recall: " + ', '.join([">%g:%g/%g" % (threshold, recall[:, i].mean(), recall[-1, i])
                                  for i, threshold in enumerate(self.threshold)]))
        logging.info("   f1_score: " + ', '.join([">%g:%g/%g" % (threshold, f1[:, i].mean(), f1[-1, i])
                                  for i, threshold in enumerate(self.threshold)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self.threshold)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self.threshold)]))
        logging.info("   GSS: " + ', '.join([">%g:%g/%g" % (threshold, gss[:, i].mean(), gss[-1, i])
                                             for i, threshold in enumerate(self.threshold)]))
        logging.info("   HSS: " + ', '.join([">%g:%g/%g" % (threshold, hss[:, i].mean(), hss[-1, i])
                                             for i, threshold in enumerate(self.threshold)]))
        logging.info("   MSE: %g/%g" % (mse.mean(), mse[-1]))
        logging.info("   MAE: %g/%g" % (mae.mean(), mae[-1]))

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