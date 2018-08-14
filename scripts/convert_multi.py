#!/usr/bin python3
""" The script to run the convert process of faceswap """

import re
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from scripts.fsmedia import Alignments, Images, Faces, Utils
from scripts.extract import Extract
from lib.utils import BackgroundGenerator, get_folder, get_image_paths
from plugins.PluginLoader_multi import PluginLoader
from time import time
import numpy as np

class Convert(object):
    """ The convert process. """

    def __init__(self, arguments):
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)

        # init for models
        self.model = self.load_model()
        self.converter = self.load_converter(self.model)


    def process(self):
        """ Original & LowMem models go with Adjust or Masked converter

            Note: GAN prediction outputs a mask + an image, while other
            predicts only an image. """
        ## TODO:统计时间消耗
        start = time()
        Utils.set_verbosity(self.args.verbose)

        self.images = Images(self.args)
        self.faces = Faces(self.args)
        self.alignments = Alignments(self.args)
        self.opts = OptionalActions(self.args, self.images.input_images)

        if not self.alignments.have_alignments_file:
            self.generate_alignments()

        self.faces.faces_detected = self.alignments.read_alignments()

        middle = time()
        ## TODO:统计时间消耗
        # batch = BackgroundGenerator(self.prepare_images(), 1)
        # 获取数据
        batch = self.prepare_images()

        self.convert_batch(self.converter, items=batch)

        # for item in tqdm(batch):
            # self.convert(self.converter, item)

        Utils.finalize(self.images.images_found,
                       self.faces.num_faces_detected,
                       self.faces.verify_output)

        self.alignments.have_alignments_file = False
        self.alignments.alignments_path = None

        end = time()
        print("process img count:" + str(len(batch)))
        print("process first step:" + str(middle - start) + "秒")
        print("process second step:" + str(end - middle) + "秒")

    def generate_alignments(self):
        """ Generate an alignments file if one does not already
        exist. Does not save extracted faces """
        print('Alignments file not found. Generating at default values...')
        # init extract
        self.extract = Extract(self.args)
        self.extract.export_face = False
        self.extract.process()

    def load_model(self):
        """ Load the model requested for conversion """
        model_name = self.args.trainer
        model_dir = get_folder(self.args.model_dir)
        num_gpus = self.args.gpus

        model = PluginLoader.get_model(model_name)(self.args.person_num, model_dir, num_gpus)

        if not model.load():
            print("Model Not Found! A valid model "
                  "must be provided to continue!")
            exit(1)

        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        args = self.args
        conv = args.converter

        converter = PluginLoader.get_converter(conv)(
            model.converter(self.args.convert_target),
            trainer=args.trainer,
            blur_size=args.blur_size,
            seamless_clone=args.seamless_clone,
            sharpen_image=args.sharpen_image,
            mask_type=args.mask_type,
            erosion_kernel_size=args.erosion_kernel_size,
            match_histogram=args.match_histogram,
            smooth_mask=args.smooth_mask,
            avg_color_adjust=args.avg_color_adjust)

        return converter

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        batch_data = []
        for filename in tqdm(self.images.input_images, file=sys.stdout):
            if not self.check_alignments(filename):
                continue
            image = Utils.cv2_read_write('read', filename)
            faces = self.faces.get_faces_alignments(filename, image)
            if not faces:
                continue
            batch_data.append([filename,image,faces])
            # yield filename, image, faces
        return batch_data

    def check_alignments(self, filename):
        """ If we have no alignments for this image, skip it """
        have_alignments = self.faces.have_face(filename)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(os.path.basename(filename)))
        return have_alignments

    def convert_batch(self, converter, items, multi_process_num = 8):
        # try:
        # face2idx
        face_all = []
        image_all = []
        filename_all =[]
        # 如果同一画面中出现多个人脸，只取面积最大的一张脸
        for idx, item in enumerate(items):
            filename_all.append(item[0])
            max_ = 0
            best_face = None
            for _, face in item[-1]:
                if face.image.shape[0]*face.image.shape[1]>max_:
                    max_ = face.image.shape[0]*face.image.shape[1]
                    best_face = face
            face_all.append(best_face)
            image_all.append(self.images.rotate_image(item[1], best_face.r))

        # convert faces by batch
        n_batch = len(face_all) // self.args.batch_size
        if len(face_all) % self.args.batch_size != 0:
            n_batch += 1
        size = 128 if (self.args.trainer.strip().lower() in ('gan128', 'originalhighres')) else 64
        new_faces = []
        mats = []
        for i in tqdm(range(n_batch)):
            face_temp = face_all[i * self.args.batch_size:(i + 1) * self.args.batch_size]
            img_temp = image_all[i * self.args.batch_size:(i + 1) * self.args.batch_size]
            # converting by model
            new_face_temp, mat_temp = converter.get_new_faces_batch(img_temp, face_temp, size)
            new_faces.append(new_face_temp)
            mats.extend(mat_temp)
        new_faces = np.concatenate(new_faces, axis=0)
        assert len(new_faces)==len(image_all)

        converter.get_new_imgs_batch([image_all, face_all, new_faces, mats, filename_all, size, self.output_dir, self.images])
        # # 多进程
        # temp_len = len(new_faces) // multi_process_num
        # pool_inputs = []
        # for i in range(len(new_faces)):
        #     pool_inputs.append([image_all[i*temp_len:(i+1)*temp_len], face_all[i*temp_len:(i+1)*temp_len],
        #                         new_faces[i*temp_len:(i+1)*temp_len,::], mats[i*temp_len:(i+1)*temp_len],
        #                         filename_all[i*temp_len:(i+1)*temp_len], size, self.output_dir, self.images])
        # pool = Pool(multi_process_num)
        # for pi in pool_inputs:
        #     pool.apply_async(converter.get_new_imgs_batch, kwds={'inputs': pi})
        # pool.close()
        # pool.join()


        # except Exception as err:
        #     print("Failed to convert images batch with", err)

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            filename, image, faces = item
            # skip = self.opts.check_skipframe(filename)
            # if not skip:
            for idx, face in faces:
                image = self.convert_one_face(converter,(filename, image, idx, face))

            # if skip != "discard":
            filename = str(self.output_dir / Path(filename).name)
            Utils.cv2_read_write('write', filename, image)
        except Exception as err:
            print("Failed to convert image: {}. "
                  "Reason: {}".format(filename, err))

    def convert_one_face(self, converter, imagevars):
        """ Perform the conversion on the given frame for a single face """
        filename, image, idx, face = imagevars

        if self.opts.check_skipface(filename, idx):
            return image

        image = self.images.rotate_image(image, face.r)
        # TODO: This switch between 64 and 128 is a hack for now.
        # We should have a separate cli option for size

        size = 128 if (self.args.trainer.strip().lower()
                       in ('gan128', 'originalhighres')) else 64

        image = converter.patch_image(image,
                                      face,
                                      size)
        image = self.images.rotate_image(image, face.r, reverse=True)
        return image


class OptionalActions(object):
    """ Process the optional actions for convert """

    def __init__(self, args, input_images):
        self.args = args
        self.input_images = input_images

        self.faces_to_swap = self.get_aligned_directory()

        self.frame_ranges = self.get_frame_ranges()
        self.imageidxre = re.compile(r"[^(mp4)](\d+)(?!.*\d)")

    # SKIP FACES #
    def get_aligned_directory(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped """
        faces_to_swap = None
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            print("Aligned directory not specified. All faces listed in the "
                  "alignments file will be converted")
        elif not os.path.isdir(input_aligned_dir):
            print("Aligned directory not found. All faces listed in the "
                  "alignments file will be converted")
        else:
            faces_to_swap = [Path(path)
                             for path in get_image_paths(input_aligned_dir)]
            if not faces_to_swap:
                print("Aligned directory is empty, "
                      "no faces will be converted!")
            elif len(faces_to_swap) <= len(self.input_images) / 3:
                print("Aligned directory contains an amount of images much "
                      "less than the input, are you sure this is the right "
                      "directory?")
        return faces_to_swap

    # SKIP FRAME RANGES #
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            return None

        minmax = {"min": 0,  # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q),
                         v.split("-")))
               for v in self.args.frame_ranges]
        return rng

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1],
                                self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe

    def check_skipface(self, filename, face_idx):
        """ Check whether face is to be skipped """
        if self.faces_to_swap is None:
            return False
        face_name = "{}_{}{}".format(Path(filename).stem,
                                     face_idx,
                                     Path(filename).suffix)
        face_file = Path(self.args.input_aligned_dir) / Path(face_name)
        skip_face = face_file not in self.faces_to_swap
        if skip_face:
            print("face {} for frame {} was deleted, skipping".format(
                face_idx, os.path.basename(filename)))
        return skip_face
