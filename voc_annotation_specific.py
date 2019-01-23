import xml.etree.ElementTree as ET
from os import getcwd
import argparse

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["car", "person", "bicycle"]

parser = argparse.ArgumentParser(description='Get file absulte ')
parser.add_argument('anno_txt', help='Path to src path.')
parser.add_argument('list_files', help='Path to save file name.')

def convert_annotation(xml_path, list_file):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if obj.find('difficult'):
            difficult = obj.find('difficult').text
            if cls not in classes or int(difficult)==1:
                continue
        else:
            if cls not in classes:
                continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def main(anno_txt, list_files):
    with open(anno_txt, 'r') as fp:
        list_file = open(list_files, 'w')
        for line in fp.readlines():
            xml_path = line.rstrip()
            img_path = xml_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
            list_file.write(img_path)
            convert_annotation(xml_path, list_file)
            list_file.write('\n')
        list_file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    anno_txt = args.anno_txt
    list_files = args.list_files
    main(anno_txt, list_files)