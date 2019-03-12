# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

import contextlib2
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def get_rect_from_file(path):
    f = open(path)
    # no use
    line = f.readline().strip()
    # get rect num
    line = f.readline().strip().split()[1]
    rect_num = int(line)
    rects = np.zeros((rect_num, 5))
    # no use
    line = f.readline().strip()
    
    # get width and height
    tmp = f.readline().strip().split()
    width = int(tmp[0])
    height = int(tmp[1])
    
    count = 0
    for i in range(rect_num):
        obj_class = int(f.readline().strip())
        line = f.readline().strip().split()
        if obj_class != 0:
            continue
        lxt = float(line[0]) if float(line[0]) >= 0 else 0;
        lyt = float(line[1]) if float(line[1]) >= 0 else 0;
        rxb = float(line[2]) if float(line[2]) < width else width-1;
        ryb = float(line[3]) if float(line[3]) < height else height-1;
        rects[count] = [obj_class, lxt, lyt, rxb, ryb]
        count += 1
 
    return rects,width, height
    
def dict_to_tf_example(img_path,box_path,label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  
  rects,width,height = get_rect_from_file(box_path)

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []
  class_name_list = ['Face']
  for rect in rects:
      xmins.append(rect[1] / width)
      ymins.append(rect[2] / height)
      xmaxs.append(rect[3] / width)
      ymaxs.append(rect[4] / height)
      class_name = class_name_list[int(rect[0])]
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example
  
def box_file_path_get(img_path):
    tmps = img_path.split('/')
    name = tmps[-1].split('.')
    name[-1] = 'box'
    tmps[-1] = '.'.join(name)
    return '/'.join(tmps)
    
def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))

      image_path = os.path.join(image_dir, example)
      if not os.path.exists(image_path):
        logging.warning('Could not find %s, ignoring example.', image_path)
        continue
      
      box_path = box_file_path_get(image_path)
      if not os.path.exists(box_path):
        logging.warning('Could not find %s, ignoring example.', box_path)
        continue

      try:
        tf_example = dict_to_tf_example(
            image_path,
            box_path,
            label_map_dict)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', image_path)

def is_img(ext):
    ext = ext.lower()
    if ext == 'jpg':
        return True
    elif ext == 'png':
        return True
    elif ext == 'jpeg':
        return True
    elif ext == 'bmp':
        return True
    else:
        return False
    
def get_img_names_from_dir(dir_name):
    files = os.listdir(dir_name)
    img_files = []
    for f in files:
        # check if it is image
        if is_img(f.split('.')[-1]) == False:
            continue
        # check box is exist
        file_path = os.path.join(dir_name, f)
        box_file_path = box_file_path_get(file_path)
        
        if os.path.exists(box_file_path) == False:
            print "Box file not exist : {}".format(box_file_path)
            continue
            
        img_files.append(f)
    return img_files
        
# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from face dataset.')
  
  ## read all images
  examples_list = get_img_names_from_dir(data_dir)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.8 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'multi_obj_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'multi_obj_val.record')

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      data_dir,
      train_examples)
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      data_dir,
      val_examples)


if __name__ == '__main__':
  tf.app.run()
