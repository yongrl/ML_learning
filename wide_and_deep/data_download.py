from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import sys

from six.moves import urllib
import tensorflow as tf


DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

## 运行时参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default='data/',help='Directory to download census data')

# 定义私有函数
def _download_and_clean_file(filename,url):
    temp_file, headers = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file,'r') as temp_eval_file:
        with tf.gfile.Open(filename,'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1]=='.':
                    line = line[:-1]
                line +='\n'
                eval_file.write(line)
    tf.gfile.Remove(temp_file)

## tf.app.run()运行时先解析了运行时参数，然后调用main(argv),将参数传递给main主体
## 所以此时定义的main函数必须有个参数
def main(_):
    # if __name__ == '__main__':只是做了一个判断，里面的变量实质上是全局的，所以这里的FLAGS是个全局变量，
    # 并不是tf.app.flags.FLAGS。
    print('FLAGS:',FLAGS)
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    # train_file_path = os.path.join(FLAGS.data_dir,TRAINING_FILE)
    # _download_and_clean_file(train_file_path,TRAINING_URL)
    #
    # eval_file_path = os.path.join(FLAGS.data_dir,EVAL_FILE)
    # _download_and_clean_file(eval_file_path,EVAL_URL)

if __name__ == '__main__':

    # namespace, args = self._parse_known_args(args, namespace)
    FLAGS, unparsed = parser.parse_known_args()
    argv=[sys.argv[0]] + unparsed
    print('argv:',argv)
    tf.app.run()

#_sys.modules['__main__'].main