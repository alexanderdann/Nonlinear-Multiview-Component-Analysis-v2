import os
import tensorflow as tf
tf.random.set_seed(3333)
from tqdm import tqdm
from sklearn.utils import gen_batches
from nmca_model import build_nmca_model, compute_loss
from TwoChannelModel import TwoChannelModel
from correlation_analysis import CCA
from tensorboard_utillities import write_scalar_summary, write_image_summary, write_PCC_summary, write_gradients_summary_mean, write_poly
from tensorboard_utillities import create_grid_writer

def generate_test_data(samples, correlations):
    data_model = TwoChannelModel(num_samples=samples)
    return data_model('Gaussian', correlations)


def train_model_v1(data, shared_dim, hidden_dim, learning_rate=1e-4, epochs=100000):
    y_1, y_2, Az_1, Az_2, z_1, z_2 = data
    lambda_reg = 1e-10
    lambda_cmplx = 0

    LOGPATH = f'{os.getcwd()}/LOG/Example'
    writer = nmca_model.create_grid_writer(LOGPATH, params=['Results', shared_dim])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model = nmca_model.build_nmca_model(hidden_dim)

    gradients_history = list()
    for epoch in tqdm(range(epochs), desc='Epochs'):
        with tf.GradientTape() as tape:
            # Watch the input to be able to compute the gradient later
            tape.watch([y_1, y_2])
            # Forward path
            [fy_1, fy_2], [yhat_1, yhat_2] = model([tf.transpose(y_1), tf.transpose(y_2)])
            # Loss computation
            loss, cca_loss, rec_loss, ccor, cmplx_losses = nmca_model.compute_loss(y_1, y_2, fy_1, fy_2, yhat_1, yhat_2,
                                                                                   shared_dim,
                                                                                   lambda_reg=lambda_reg,
                                                                                   lambda_cmplx=lambda_cmplx)
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients_history.append(gradients)
            # Backpropagate through network
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 10 == 0:
                B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, 5)
                sim_v2 = nmca_model.compute_similarity_metric_v1(S=z_1[:2], U=epsilon)
                sim_v3 = nmca_model.compute_similarity_metric_v1(S=z_2[:2], U=omega)
                dist = nmca_model.compute_distance_metric(S=z_1[:2], U=0.5 * (omega + epsilon)[:2])
                l1, rademacher = nmca_model.compute_rademacher(model)
                write_gradients_summary_mean(writer=writer, gradients=gradients_history, epoch=epoch,
                                             trainable_variables=model.trainable_variables)

                write_scalar_summary(
                    writer=writer,
                    epoch=epoch,
                    list_of_tuples=[
                        (cmplx_losses[0], 'Polynomial Loss/View 1'),
                        (cmplx_losses[1], 'Polynomial Loss/View 2'),
                        (cmplx_losses[2], 'Polynomial Loss/Total'),
                        (loss, 'Loss/Total'),
                        (cca_loss, 'Loss/CCA'),
                        (rec_loss, 'Loss/Reconstruction'),
                        (ccor[0], 'Canonical correlation/1'),
                        (ccor[1], 'Canonical correlation/2'),
                        (ccor[2], 'Canonical correlation/3'),
                        (ccor[3], 'Canonical correlation/4'),
                        (ccor[4], 'Canonical correlation/5'),
                        (dist, 'Performance Measures/Distance measure'),
                        (sim_v2, 'Performance Measures/Similarity measure 1st view'),
                        (sim_v3, 'Performance Measures/Similarity measure 2nd view'),
                        (l1, 'Regularisation Measures/L1'),
                        (rademacher, 'Regularisation Measures/Rademacher'),
                    ]
                )
                # Resetting the history of gradients
                gradients_history = list()

            if epoch % 2500 == 0 or epoch == 99999:
                write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2, yhat_1, yhat_2)
                write_PCC_summary(writer, epoch, z_1, z_2, epsilon, omega, 1000)


def train_model_v2(data, batch_size, shared_dim, hidden_dim, pca_dim, desc, hidden_layers=1, learning_rate=1e-4, epochs=100000):
    channels, samples = data[0].shape
    num_batches = samples//batch_size

    tmp_1, tmp_2 = list(), list()
    for batch_idx in gen_batches(samples, batch_size):
        tmp_1.append(data[0][:, batch_idx].T)
        tmp_2.append(data[1][:, batch_idx].T)
    y_1 = tf.convert_to_tensor(tmp_1, dtype=tf.float32)
    y_2 = tf.convert_to_tensor(tmp_2, dtype=tf.float32)

    lambda_reg = 1e-10
    lambda_cmplx = 0

    LOGPATH = f'{os.getcwd()}/LOG/{desc}'
    MODELSPATH = f'{os.getcwd()}/MODELS/{desc}'
    writer = create_grid_writer(root_dir=LOGPATH, params=['Shared Dim', shared_dim, 'Batch Size', batch_size])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model = build_nmca_model(hidden_dim=hidden_dim, channels=channels, hidden_layers=hidden_layers)

    for epoch in tqdm(range(epochs), desc='Epochs'):
        losses, cca_losses, rec_losses = list(), list(), list()
        intermediate_outputs = list()
        for batch_idx in range(num_batches):
            batch_y1, batch_y2 = y_1[batch_idx], y_2[batch_idx]

            with tf.GradientTape() as tape:
                # Watch the input to be able to compute the gradient later
                tape.watch([batch_y1, batch_y2])

                # Forward path
                [fy_1, fy_2], [yhat_1, yhat_2] = model([batch_y1, batch_y2])
                # Loss computation
                loss, cca_loss, rec_loss, ccor, cmplx_losses = compute_loss(batch_y1, batch_y2,
                                                                                fy_1, fy_2,
                                                                                yhat_1, yhat_2,
                                                                                shared_dim, pca_dim,
                                                                                lambda_reg, lambda_cmplx)
                losses.append(loss)
                cca_losses.append(cca_loss)
                rec_losses.append(rec_loss)
                intermediate_outputs.append((fy_1, fy_2))

                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                # Backpropagate through network
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 25 == 0:
            tmp = list()
            for batch_idx in range(num_batches):
                batched_fy_1, batched_fy_2 = intermediate_outputs[batch_idx]
                B1, B2, epsilon, omega, ccor = CCA(batched_fy_1, batched_fy_2, shared_dim, pca_dim)
                tmp.append(ccor)

            avg_ccor = tf.math.reduce_mean(tmp, axis=0)
            static_part = [(cmplx_losses[0], 'Polynomial Loss/View 1'),
                           (cmplx_losses[1], 'Polynomial Loss/View 2'),
                           (cmplx_losses[2], 'Polynomial Loss/Total'),
                           (tf.math.reduce_mean(losses), 'Loss/Total'),
                           (tf.math.reduce_mean(cca_losses), 'Loss/CCA'),
                           (tf.math.reduce_mean(rec_losses), 'Loss/Reconstruction')]

            dynamic_part = [(cval, f'Canonical correlation/{idx})') for idx, cval in enumerate(avg_ccor)]
            write_scalar_summary(
                writer=writer,
                epoch=epoch,
                list_of_tuples=static_part + dynamic_part
            )

    try:
        os.makedirs(MODELSPATH)
    except FileExistsError:
        print('MODELS PATH exists, saving data.')
    finally:
        model.save(f'{MODELSPATH}/SharedDim-{shared_dim}-BatchSize-{batch_size}.tf',
                   overwrite=False)
