import os
import numpy as np
import tensorflow as tf
import scipy
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from correlation_analysis import CCA


def _build_channel_model(channel, view_idx, channel_idx, hidden_dim, hidden_layers):
    # Build encoder
    enc_input = tf.keras.layers.Dense(
        hidden_dim,
        activation='relu',
        name=f'View_{view_idx}_Encoder_DenseLayer_Channel_{channel_idx}'
    )(channel)
    enc_output = tf.keras.layers.Dense(
        1,
        activation=None,
        name=f'View_{view_idx}_Encoder_OutputLayer_Channel_{channel_idx}'
    )(enc_input)

    # Build decoder
    dec_input = tf.keras.layers.Dense(
        hidden_dim,
        activation='relu',
        name=f'View_{view_idx}_Decoder_DenseLayer_Channel_{channel_idx}'
    )(enc_output)
    dec_output = tf.keras.layers.Dense(
        1,
        activation=None,
        name=f'View_{view_idx}_Decoder_OutputLayer_Channel_{channel_idx}'
    )(dec_input)

    return enc_output, dec_output


def _build_view_models(inputs, view_idx, hidden_dim, hidden_layers):
    enc_outputs = list()
    dec_outputs = list()

    for idx, channel in enumerate(inputs):
        enc_output, dec_output = _build_channel_model(channel, view_idx, idx, hidden_dim, hidden_layers)
        enc_outputs.append(enc_output)
        dec_outputs.append(dec_output)

    enc_outputs = tf.keras.layers.concatenate(
        enc_outputs,
        name=f'View_{view_idx}_Encoder_OutputConcatenation'
    )
    dec_outputs = tf.keras.layers.concatenate(
        dec_outputs,
        name=f'View_{view_idx}_Decoder_OutputConcatenation'
    )

    return enc_outputs, dec_outputs


def build_nmca_model(hidden_dim, channels=5, hidden_layers=1):
    # Please mind that sklearn and tensorflow interpret batch_size differently.
    # We interpret it as samples per batch and not total amount of batches
    inp_view_1 = tf.keras.layers.Input(shape=(channels,))
    inp_view_2 = tf.keras.layers.Input(shape=(channels,))
    view_1_splits = tf.split(inp_view_1, num_or_size_splits=channels, axis=1)
    view_2_splits = tf.split(inp_view_2, num_or_size_splits=channels, axis=1)

    enc_outputs_1, dec_outputs_1 = _build_view_models(view_1_splits, 0, hidden_dim, hidden_layers)
    enc_outputs_2, dec_outputs_2 = _build_view_models(view_2_splits, 1, hidden_dim, hidden_layers)

    model = tf.keras.Model(
        inputs=[inp_view_1, inp_view_2],
        outputs=[[enc_outputs_1, enc_outputs_2], [dec_outputs_1, dec_outputs_2]]
    )

    return model


def compute_loss(y_1, y_2, fy_1, fy_2, yhat_1, yhat_2, shared_dim, pca_dim, lambda_reg=0.001,  lambda_cmplx=0.1):
    # CCA loss
    NUM_SAMPLES = y_1.numpy().shape[1]
    B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, shared_dim, pca_dim)
    cca_loss = tf.reduce_mean(tf.square(tf.norm(tf.subtract(epsilon, omega), axis=0))) / shared_dim

    # Reconstruction loss
    rec_loss_1 = tf.square(tf.norm(y_1 - yhat_1, axis=0))
    rec_loss_2 = tf.square(tf.norm(y_2 - yhat_2, axis=0))
    rec_loss = tf.reduce_mean(tf.add(rec_loss_1, rec_loss_2))

    # Complexity loss
#   residuals = np.empty(shape=(2, 5))
#   degree = 3

#   std_fy_1 = tf.math.reduce_std(fy_1, 0)[None]
#   std_fy_2 = tf.math.reduce_std(fy_2, 0)[None]

#   norm_fy_1 = tf.transpose(fy_1 / tf.tile(std_fy_1, tf.constant([NUM_SAMPLES, 1], tf.int32)))
#   norm_fy_2 = tf.transpose(fy_2 / tf.tile(std_fy_2, tf.constant([NUM_SAMPLES, 1], tf.int32)))

#   for idx in range(tf.shape(y_1)[0]):
#       coeff1, diagnostic1 = Polynomial.fit(y_1[idx].numpy(), norm_fy_1[idx].numpy(), degree, full=True)
#       residuals[0, idx] = diagnostic1[0][0]

#       coeff2, diagnostic2 = Polynomial.fit(y_2[idx].numpy(),  norm_fy_2[idx].numpy(), degree, full=True)
#       residuals[1, idx] = diagnostic2[0][0]


#   residuals_tf = tf.convert_to_tensor(residuals, dtype=tf.float32)

#   closs_1 = tf.math.reduce_euclidean_norm(residuals_tf[0], axis=0)
#   closs_2 = tf.math.reduce_euclidean_norm(residuals_tf[1], axis=0)

#   cmplx_loss = tf.reduce_mean([
#                   tf.math.reduce_euclidean_norm(residuals_tf[0], axis=0),
#                   tf.math.reduce_euclidean_norm(residuals_tf[1], axis=0)
#                               ])

    # Combine losses
    loss = cca_loss + lambda_reg * rec_loss #+ lambda_cmplx * cmplx_loss

    return loss, cca_loss, rec_loss, ccor, (0, 0, 0)#(closs_1, closs_2, cmplx_loss)