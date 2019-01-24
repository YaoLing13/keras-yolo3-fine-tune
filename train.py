"""
Retrain the YOLO model for your own dataset.
"""
import xml.etree.ElementTree as ET
import os
import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

parser = argparse.ArgumentParser(description='Get config file path')
parser.add_argument('config_path', nargs='?', default="config/config.xml", help='Path to src path.')


def _main():
    annotation_path = 'dataset/train_label.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes_specific.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    pretrain_model_path = 'model_data/yolo_weights.h5'
    trainging_model_filename = log_dir + "weights.best.h5"  # yl-0124
    save_model_stage_name = log_dir + 'trained_weights_stage_1.h5'
    save_model_final_name = log_dir + 'trained_weights_final.h5'
    input_image_height = 416
    input_image_width = 416
    ## train params
    batch_size_stage1 = 32
    initial_epochs_stage1 = 0
    epochs_stage1 = 50
    batch_size_stage2 = 32
    initial_epochs_stage2 = epochs_stage1
    epochs_stage2 = 100
    is_retrain = False

    root_path = os.getcwd()
    print('#### root_path : %s ####' % root_path)
    args = parser.parse_args()
    config_path = args.config_path
    print('#### config_path: %s ####' % config_path)
    with open(config_path) as fp:
        print('-------- Using Config file params --------')
        tree = ET.parse(fp)
        root = tree.getroot()
        for obj in root.iter('train'):
            annotation_path = root_path + '/' + obj.find('annotation_path').text
            log_dir = root_path + '/' + obj.find('log_dir').text
            classes_path = root_path + '/' + obj.find('classes_path').text
            anchors_path = root_path + '/' + obj.find('anchors_path').text
            pretrain_model_path = root_path + '/' + obj.find('pretrain_model_path').text
            trainging_model_filename = log_dir + obj.find('trainging_model_filename').text
            save_model_stage_name = log_dir +  obj.find('save_model_stage_name').text
            save_model_final_name = log_dir +  obj.find('save_model_final_name').text
            print('*****************************************************************************************')
            print('**** model params ****')
            print('annotation_path         : %s' % annotation_path)
            print('log_dir                 : %s' % log_dir)
            print('classes_path            : %s' % classes_path)
            print('anchors_path            : %s' % anchors_path)
            print('pretrain_model_path     : %s' % pretrain_model_path)
            print('trainging_model_filename: %s' % trainging_model_filename)
            print('save_model_stage_name   : %s' % save_model_stage_name)
            print('save_model_final_name   : %s' % save_model_final_name)

            input_image_height = int(obj.find('input_image_height').text)
            input_image_width = int(obj.find('input_image_width').text)

            batch_size_stage1 = int(obj.find('batch_size_stage1').text)
            initial_epochs_stage1 = int(obj.find('initial_epochs_stage1').text)
            epochs_stage1 = int(obj.find('epochs_stage1').text)
            is_retrain = bool(obj.find('is_retrain').text)
            batch_size_stage2 = int(obj.find('batch_size_stage2').text)
            initial_epochs_stage2 = int(obj.find('initial_epochs_stage2').text)
            epochs_stage2 = int(obj.find('epochs_stage2').text)
            print('*****************************************************************************************')
            print('**** train params ****')
            print('input_image_height   : %d' %input_image_height)
            print('input_image_width    : %d' %input_image_width)
            print('-----------------------------------------------')
            print('retrain              : %d' %is_retrain)
            print('-----------------------------------------------')
            print('batch_size_stage1    : %d' %batch_size_stage1)
            print('initial_epochs_stage1: %d' %initial_epochs_stage1)
            print('epochs_stage1        : %d' %epochs_stage1)
            print('-----------------------------------------------')
            print('batch_size_stage2    : %d' %batch_size_stage2)
            print('initial_epochs_stage2: %d' %initial_epochs_stage2)
            print('epochs_stage2        : %d' %epochs_stage2)
            print('*****************************************************************************************')

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (input_image_height,input_image_width) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            # freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
            freeze_body=2, weights_path=pretrain_model_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
            # freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
            freeze_body=2, weights_path=pretrain_model_path) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    # checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    checkpoint = ModelCheckpoint(trainging_model_filename, # yl-0124
        monitor='val_loss', save_weights_only=True, save_best_only=True, mode='max', period=3) # yl-0124
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if not is_retrain:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = batch_size_stage1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs_stage1,
                initial_epoch=initial_epochs_stage1,
                callbacks=[logging, checkpoint])
        model.save_weights(save_model_stage_name)

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if is_retrain:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = batch_size_stage2 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs_stage2,
            initial_epoch=initial_epochs_stage2,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(save_model_final_name)

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
