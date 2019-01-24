import xml.etree.ElementTree as ET
import os

if __name__ == '__main__':
    path = 'config/config.xml'
    root_path = os.getcwd()
    print('#### root_path: %s ####' % root_path)
    with open(path) as fp:
        print('-------- Using Config file params --------')
        tree = ET.parse(fp)
        root = tree.getroot()
        for obj in root.iter('train'):
            annotation_path = root_path + '/' + obj.find('annotation_path').text
            log_dir = root_path + '/' + obj.find('log_dir').text
            classes_path = root_path + '/' + obj.find('classes_path').text
            anchors_path = root_path + '/' + obj.find('anchors_path').text
            pretrain_model_path = root_path + '/' + obj.find('pretrain_model_path').text
            trainging_model_filename = root_path + '/' + obj.find('trainging_model_filename').text
            save_model_stage_name = root_path + '/' + obj.find('save_model_stage_name').text
            save_model_final_name = root_path + '/' + obj.find('save_model_final_name').text
            print('**************************************************')
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
            batch_size_stage2 = int(obj.find('batch_size_stage2').text)
            print('**************************************************')
            print('**** train params ****')
            print('input_image_height   : %d' %input_image_height)
            print('input_image_width    : %d' %input_image_width)
            print('batch_size_stage1    : %d' %batch_size_stage1)
            print('initial_epochs_stage1: %d' %initial_epochs_stage1)
            print('epochs_stage1        : %d' %epochs_stage1)
            print('batch_size_stage2    : %d' %batch_size_stage2)

