import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss


# ORIGINAL: https://github.com/GB-TonyLiang/DCA/blob/master/loss_fn.py
# Derived from https://github.com/mdca-loss/MDCA-Calibration/blob/main/solvers/loss.py

from keras.losses import CategoricalCrossentropy
from keras.activations import softmax

def cross_entropy_with_dca_loss(logits, labels, weights=None, alpha=1., beta=10.):
    cce = CategoricalCrossentropy() 
    ce = cce(logits, labels) # not sure about weights parameter. #
    softmaxes = softmax(logits, axis=1)
    confidences = tf.reduce_max(softmaxes, axis=1)
    mean_conf = tf.reduce_mean(confidences)
    #labels [8,2], logits[8,2]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmaxes,axis=1), tf.argmax(labels,axis=1)), dtype=tf.float32))
    dca = tf.abs(mean_conf - acc)
    loss = alpha * ce + beta * dca
    return loss


# https://github.com/mdca-loss/MDCA-Calibration/blob/main/solvers/loss.py
# Change the classes for each dataset you use
def cross_entropy_with_mdca_loss(logits, labels, alpha = 0.1, beta = 1.0, gamma = 1.0,classes=2):
    cce = CategoricalCrossentropy() 
    ce = cce(logits, labels) # not sure about weights parameter.
    softmaxes = softmax(logits, axis=1)
    loss = tf.constant(0.0, dtype=tf.float32) 
    for c in range(classes):
        avg_count = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,axis=1), c), dtype=tf.float32))
        avg_conf = tf.reduce_mean(softmaxes[:, c])
        loss += tf.abs(avg_conf - avg_count)
    denom = classes
    loss /= denom
    mdca = alpha * ce + beta * loss
    return mdca



class ECELoss(tf.keras.layers.Layer):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def call(self, logits, labels):
        softmaxes = tf.nn.softmax(logits, axis=1)
        confidences = tf.reduce_max(softmaxes, axis=1)
        predictions = tf.argmax(softmaxes, axis=1, output_type=tf.dtypes.int32)
        accuracies = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)

        ece = tf.constant(0.0, dtype=tf.float32)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = tf.math.logical_and(
                tf.math.greater(confidences, bin_lower),
                tf.math.less_equal(confidences, bin_upper)
            )
            prop_in_bin = tf.reduce_mean(tf.cast(in_bin, dtype=tf.float32))
            if prop_in_bin > 0:
                accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(accuracies, in_bin))
                avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
                ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    


class AdaptiveECELoss(tf.keras.layers.Layer):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1), np.arange(npt), np.sort(x))
    
    def call(self, logits, labels):
        softmaxes = tf.nn.softmax(logits, axis=1)
        confidences = tf.reduce_max(softmaxes, axis=1)
        predictions = tf.argmax(softmaxes, axis=1, output_type=tf.dtypes.int32)
        
        accuracies = tf.cast(tf.equal(predictions,labels), dtype=tf.float32)
        
        n, bin_boundaries = np.histogram(confidences, self.histedges_equalN(confidences))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = tf.constant(0.0, dtype=tf.float32)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = tf.math.logical_and(
                tf.math.greater(confidences, bin_lower),
                tf.math.less_equal(confidences, bin_upper)
            )
            prop_in_bin = tf.reduce_mean(tf.cast(in_bin, dtype=tf.float32))
            if prop_in_bin > 0:
                accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(accuracies, in_bin))
                avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
                ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ClasswiseECELoss(tf.keras.layers.Layer):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def call(self, logits, labels):
        num_classes = int(tf.reduce_max(labels).numpy()) + 1
        softmaxes = tf.nn.softmax(logits, axis=1)
        per_class_sce = []

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = tf.constant(0.0, dtype=tf.float32)
            labels_in_class = tf.math.equal(labels, i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = tf.math.logical_and(
                tf.math.greater(class_confidences, bin_lower),
                tf.math.less_equal(class_confidences, bin_upper)
            )
                prop_in_bin = tf.reduce_mean(tf.cast(in_bin, dtype=tf.float32))

                if prop_in_bin > 0:
                    
                    # accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(labels_in_class, in_bin))
                    accuracy_in_bin = tf.reduce_mean(tf.cast(tf.boolean_mask(labels_in_class, in_bin), dtype=tf.float32))
                    avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(class_confidences, in_bin))
                    class_sce += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            per_class_sce.append(class_sce)

        sce = np.array(per_class_sce).mean()
        return sce
    
    
def ClasswiseAccuracy(testLabels, t):
    # Calculate accuracy for each class
    # Identify unique class labels
    unique_classes = np.unique(testLabels)
    # Initialize a dictionary to store class-wise counts
    class_counts = {class_label: 0 for class_label in unique_classes}
    # Count the total number of examples for each class
    for class_label in testLabels:
        class_counts[class_label] += 1
    # Initialize a dictionary to store class-wise correct counts
    class_correct_counts = {class_label: 0 for class_label in unique_classes}
    # Calculate the number of correct classifications for each class
    count_correct = 0
    for true_label, predicted_label in zip(testLabels, t):
        if true_label == predicted_label:
            count_correct=count_correct+1
            class_correct_counts[true_label] += 1
    # print("Total Correct Count",count_correct)
    # print("Each class correct count",class_correct_counts)
    # print("Each class count ",class_counts)
    # Calculate class-wise accuracy by dividing correct counts by total counts
    class_accuracies = {class_label: correct_count / total_count for class_label, correct_count, total_count in zip(unique_classes, class_correct_counts.values(), class_counts.values())}
    return class_accuracies


# KERAS Implementatoin of the pytorch code below.
# Paper: Calibrating Deep Neural Networks using Focal Loss - J Mukhoti  (Neurips 2020)
# IMPLEMENTATION CODE: https://github.com/torrvision/focal_calibration/blob/main/Losses/brier_score.py
class BrierScore(tf.keras.layers.Layer):
    def __init__(self):
        super(BrierScore, self).__init__()

    def call(self, inputs, targets):
        if tf.rank(inputs) > 2:
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], -1))# N,C,H,W => N,C,H*W
            inputs = tf.transpose(inputs, perm=(0, 2, 1))# N,C,H*W => N,H*W,C
            inputs = tf.reshape(inputs, (-1, tf.shape(inputs)[2]))# N,H*W,C => N*H*W,C

        # targets = tf.reshape(targets, (-1, 1))
        # target_one_hot = tf.zeros(tf.shape(inputs))
        # target_one_hot = tf.tensor_scatter_nd_add(target_one_hot, tf.concat([tf.range(tf.shape(targets)[0])[:, tf.newaxis], targets], axis=1), 1)

        pt = tf.nn.softmax(inputs, axis=1)
        squared_diff = tf.square(targets - pt)

        loss = tf.reduce_sum(squared_diff) / tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        return loss
    
    
    
class ReliabilityECELoss(tf.keras.layers.Layer):
    def __init__(self, n_bins=15):
        super(ReliabilityECELoss, self).__init__()
        bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def call(self, logits, labels):
        softmaxes = tf.nn.softmax(logits, axis=1)
        confidences = tf.reduce_max(softmaxes, axis=1)
        predictions = tf.argmax(softmaxes, axis=1, output_type=tf.dtypes.int32)
        accuracies = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
        bin_ece_list = []
        
        ece = tf.constant(0.0, dtype=tf.float32)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = tf.math.logical_and(
                tf.math.greater(confidences, bin_lower),
                tf.math.less_equal(confidences, bin_upper)
            )
            prop_in_bin = tf.reduce_mean(tf.cast(in_bin, dtype=tf.float32))
            if prop_in_bin > 0:
                accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(accuracies, in_bin))
                avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
                bin_ece_list.append(tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin)
                ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece, bin_ece_list





import sklearn.metrics as skm

# print(skm.accuracy_score(np.argmax(testLabels, axis=1), np.argmax(t1, axis=1)))

from joblib import Parallel, delayed
from functools import partial


def accuracy_retention_curve(ground_truth, predictions, uncertainties, fracs_retained, parallel_backend=None):
        """
        Compute Accuracy retention curve.
        
        Args:
        ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                        with shape [H * W * D].
        predictions:  `numpy.ndarray`, binary segmentation predictions,
                        with shape [H * W * D].
        uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                        with shape [H * W * D].
        fracs_retained:  `numpy.ndarray`, array of increasing valies of retained 
                        fractions of most certain voxels, with shape [N].
        parallel_backend: `joblib.Parallel`, for parallel computation
                        for different retention fractions.
        Returns:
        (y-axis) nDSC at each point of the retention curve (`numpy.ndarray` with shape [N]).
        """

        def compute_acc(frac_, preds_, gts_, N_):
            pos = int(N_ * frac_)
            curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
            
            return skm.accuracy_score(gts_, curr_preds)

        if parallel_backend is None:
            parallel_backend = Parallel(n_jobs=1)

        ordering = uncertainties.argsort()
        gts = ground_truth[ordering].copy()
        preds = predictions[ordering].copy()
        
        N = len(gts)

        process = partial(compute_acc, preds_=preds, gts_=gts, N_=N)
        accuracy_scores = np.asarray(
            parallel_backend(delayed(process)(frac) for frac in fracs_retained)
        )

        return accuracy_scores
    
def save_retention_curve_values(testLabels, t1, m_name, save_dir, net):
    
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)
    n_jobs = 1

    with Parallel(n_jobs=n_jobs) as parallel_backend:
        accuracy_rc = accuracy_retention_curve(ground_truth=np.argmax(testLabels, axis=1), predictions=np.argmax(t1, axis=1), uncertainties=np.argmax(t1, axis=1), fracs_retained=fracs_retained, parallel_backend=parallel_backend)

    # print(accuracy_rc)
    print(f'Saved in: {save_dir}')
    print(net)
    if net == 'baseline':
        np.savetxt(f'{save_dir}/{m_name}_baseline.txt', accuracy_rc, delimiter=",")  
    elif net == 'LS':
        np.savetxt(f'{save_dir}/{m_name}_LS.txt', accuracy_rc, delimiter=",")    
    elif net == 'FL':
        np.savetxt(f'{save_dir}/{m_name}_FL.txt', accuracy_rc, delimiter=",")      
    elif net == 'dca':
        np.savetxt(f'{save_dir}/{m_name}_dca.txt', accuracy_rc, delimiter=",") 
    elif net == 'mdca':
        np.savetxt(f'{save_dir}/{m_name}_mdca.txt', accuracy_rc, delimiter=",")  
    elif net == 'ours_alpha05':
        np.savetxt(f'{save_dir}/{m_name}_ours.txt', accuracy_rc, delimiter=",")

