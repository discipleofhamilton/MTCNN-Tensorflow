# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import time
import os
import shutil

import tensorflow as tf
import cv2
import numpy as np

from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames, detect_face_24net, detect_face_12net

def main(args):

    detect_totalTime = 0.0
    totalTime = 0.0
    frameCount = 0


    if args.save_image:
        output_directory = args.save_path
        print(args.save_image)
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        else:
            os.mkdir(output_directory)

    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:

                file_paths = get_model_filenames(args.model_dir)

                if len(file_paths) == 3:
                    image_pnet = tf.placeholder(
                        tf.float32, [None, None, None, 3])
                    pnet = PNet({'data': image_pnet}, mode='test')
                    out_tensor_pnet = pnet.get_all_output()

                    image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                    rnet = RNet({'data': image_rnet}, mode='test')
                    out_tensor_rnet = rnet.get_all_output()

                    image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                    onet = ONet({'data': image_onet}, mode='test')
                    out_tensor_onet = onet.get_all_output()

                    saver_pnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "pnet/"])
                    saver_rnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "rnet/"])
                    saver_onet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                    if v.name[0:5] == "onet/"])

                    saver_pnet.restore(sess, file_paths[0])

                    def pnet_fun(img): return sess.run(
                        out_tensor_pnet, feed_dict={image_pnet: img})

                    saver_rnet.restore(sess, file_paths[1])

                    def rnet_fun(img): return sess.run(
                        out_tensor_rnet, feed_dict={image_rnet: img})

                    saver_onet.restore(sess, file_paths[2])

                    def onet_fun(img): return sess.run(
                        out_tensor_onet, feed_dict={image_onet: img})

                else:
                    saver = tf.train.import_meta_graph(file_paths[0])
                    saver.restore(sess, file_paths[1])

                    def pnet_fun(img): return sess.run(
                        ('softmax/Reshape_1:0',
                        'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                    def rnet_fun(img): return sess.run(
                        ('softmax_1/softmax:0',
                        'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                    def onet_fun(img): return sess.run(
                        ('softmax_2/softmax:0',
                        'onet/conv6-2/onet/conv6-2:0',
                        'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})

                for filename in os.listdir(args.image_path):

                    img = cv2.imread(os.path.join(args.image_path, filename))
                    resized_image = cv2.resize(img, (320, 180))
                    # resized_image = cv2.resize(img, (640, 360))

                    start_time = time.time()*1000

                    # P-Net + R-Net + O-Net
                    if args.net == "ALL":
                        rectangles, points = detect_face(resized_image, args.minsize,
                                                        pnet_fun, rnet_fun, onet_fun,
                                                        args.threshold, args.factor)

                    # P-Net + R-Net without faces' landmarks
                    elif args.net == "PR":
                        rectangles = detect_face_24net(resized_image, args.minsize, 
                                                        pnet_fun, rnet_fun,
                                                        args.threshold, args.factor)

                    # Only P-Net
                    elif args.net == "P":
                        rectangles = detect_face_12net(resized_image, args.minsize,
                                                        pnet_fun, args.threshold, args.factor)

                    else:
                        print("ERROR: WRONG NET INPUT")
                    end_time = time.time()*1000
                    detect_totalTime = detect_totalTime + (end_time - start_time)

                    print(filename + " time : " + str(end_time - start_time) + "ms")

                    # print(type(rectangles))
                    if args.net == "ALL":
                        points = np.transpose(points) # The outputs of O-Net which are faces' landmarks
                    for rectangle in rectangles:
                        cv2.putText(resized_image, str(rectangle[4]),
                                    (int(rectangle[0]), int(rectangle[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0))
                        cv2.rectangle(resized_image, (int(rectangle[0]), int(rectangle[1])),
                                    (int(rectangle[2]), int(rectangle[3])),
                                    (255, 0, 0), 1)

                    if args.net == "ALL":
                        for point in points:
                            for i in range(0, 10, 2):
                                cv2.circle(resized_image, (int(point[i]), int(
                                    point[i + 1])), 2, (0, 255, 0))
                    cv2.imshow("MTCNN-Tensorflow wangbm", resized_image)
                    frameCount = frameCount + 1
                    if args.save_image:
                        outputFilePath = os.path.join(output_directory, filename)
                        cv2.imwrite(outputFilePath, resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
        
    detect_average_time = detect_totalTime/frameCount
    print("detection average time: " + str(detect_average_time) + "ms" )
    print("detection fps: " + str(1/(detect_average_time/1000)))

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str,
                        help='The image path of the testing image')
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./save_model/all_in_one/')
    parser.add_argument('--net', type=str, choices=["P", "PR", "ALL"],
                        help='The minimum size of face to detect.', default="ALL")
    parser.add_argument(
        '--threshold',
        type=float,
        nargs=3,
        help='Three thresholds for pnet, rnet, onet, respectively.',
        default=[0.8, 0.8, 0.8])
    parser.add_argument('--minsize', type=int,
                        help='The minimum size of face to detect.', default=20)
    parser.add_argument('--factor', type=float,
                        help='The scale stride of orginal image', default=0.7)
    parser.add_argument('--save_image', type=bool,
                        help='Whether to save the result image', default=False)
    parser.add_argument('--save_path', type=str,
                        help='Where to save the result image', default=False)                    
    parser.add_argument('--save_name', type=str,
                        help='If save_image is true, specify the output path.',
                        default='result.jpg')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
