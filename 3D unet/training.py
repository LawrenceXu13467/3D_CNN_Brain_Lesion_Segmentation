import math
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=gettingCallbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))

def gettingCallbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(decaysteps, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks

def coef_dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def coef_dice_loss(y_true, y_pred):
    return -coef_dice(y_true, y_pred)


def w_coef_dice(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def w_coef_dice_loss(y_true, y_pred):
    return -w_coef_dice(y_true, y_pred)


def label_wise_coef_dice(y_true, y_pred, label_index):
    return coef_dice(y_true[:, label_index], y_pred[:, label_index])


def get_label_coef_dice_function(label_index):
    f = partial(label_wise_coef_dice, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = coef_dice
dice_coef_loss = coef_dice_loss

K.set_image_dim_ordering('th')



def decaysteps(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'coef_dice_loss': coef_dice_loss, 'coef_dice': coef_dice,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'w_coef_dice': w_coef_dice,
                      'w_coef_dice_loss': w_coef_dice_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

