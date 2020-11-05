# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:00:00 2020

@author: nprakash
"""

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend as K



def tversky_index(y_true, y_pred, smooth = 0.5, alpha = 0.70):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f)) 
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f) 
    TI_score = (true_pos + smooth)/(true_pos + alpha * false_neg + (1-alpha) * false_pos + smooth)
    return TI_score



def focal_tversky_loss(y_true, y_pred, gamma = 1.5, alpha = 0.6):
    TI_score = tversky_index(y_true, y_pred, alpha = alpha)
    scaledCoeff = tf.pow((TI_score), 1/gamma)
    loss = 1 - scaledCoeff
    return loss



def BCE_F_TI_LOSS(ce_weight = 0.2, gamma = 2, alpha = 0.7):
    def bce_focal_tversky_loss(y_true, y_pred):
    	loss = ce_weight * losses.binary_crossentropy(y_true, y_pred) + (1 - ce_weight) * focal_tversky_loss(y_true, y_pred, gamma = gamma, alpha = alpha)
    	return loss
    return bce_focal_tversky_loss



def MCC(y_true, y_pred):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def PRED_AREA(y_true, y_pred, trimWidth = 20):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp + fp)
    denominator = (tp + fp + fn + tn)

    return numerator / (denominator + K.epsilon())


def POD(y_true, y_pred):
    """
    Hit_Rate  prob of detection  OR  Recall  OR  sensitivity
    
    Shoud be high to avoid FP detections
    """

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    # y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    # tn = K.sum(y_neg * y_pred_neg)

    # fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp)
    denominator = (tp + fn)

    return numerator / (denominator + K.epsilon())


def POFD(y_true, y_pred):
    """
    Miss_Rate    prob of false detection   OR   Fall Out   OR  False Positive Rate
    """

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    # tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    # fn = K.sum(y_pos * y_pred_neg)

    numerator = (fp)
    denominator = (fp + tn)

    return numerator / (denominator + K.epsilon())



    
