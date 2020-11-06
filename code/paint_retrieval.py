import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial as spatial
from constants import *
from scipy.spatial.distance import cdist


class Retrieve:
    def __init__(self, vector_size=128):
        """
            Retrieve initialization
            param:
                    vector_size: the max number of keypoints to store
          """
        self.names = []
        self.matrix = []
        self.vector_size = vector_size
        self.data = None

    def mask(self, im):
        """
            Retrieve initialization
            param:
                    im: image to apply filter on.
          """
        mask = np.zeros_like(im)
        f = 0.4
        rows, cols, _ = mask.shape
        # create a white filled ellipse
        mask = cv2.ellipse(mask, center=(int(cols // 2), int(rows // 2)), axes=(int(cols * f), int(rows * f)), angle=0,
                           startAngle=0,
                           endAngle=360,
                           color=(255, 255, 255), thickness=-1)
        # Bitwise AND operation to black out regions outside the mask
        result = np.bitwise_and(im, mask)
        return result

    def get_fv(self, im):
        """
            Retrieve initialization
            param:
                    im: image to extract feature vectors and keypoints
          """
        im = cv2.resize(im, (700, 700))
        im = self.mask(im)
        alg = cv2.xfeatures2d.SIFT_create()
        kps = alg.detect(im)
        kps = sorted(kps, key=lambda x: -x.response)[:self.vector_size]
        kps, dsc = alg.compute(im, kps)
        if kps is None or dsc is None:
            return None
        dsc = dsc.flatten()
        needed_size = self.vector_size * 128
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc.astype(np.float32)

    def create_db(self, db_name=CFG_DIR + "defaultdb.pck"):
        """
            Db creation
          """
        print("Creating paintings db...")
        result = {}
        for filename in os.listdir(DATABASE_DIR):
            im = cv2.imread(DATABASE_DIR + filename)
            dsc = self.get_fv(im)
            result[filename] = dsc[:self.vector_size * 128]
            print("Processed: ", filename)
        with open(db_name, 'wb', pickle.HIGHEST_PROTOCOL) as fp:
            pickle.dump(result, fp)

    def dist(self, dsc1, st=0.4):
        """
            Score assignment
            param:
                    dsc1: image descriptor
                    st: threshold for matching
          """
        dsc1 = dsc1.reshape(self.vector_size, 128)
        result = np.zeros(self.matrix.shape[0])
        for index, elem in enumerate(self.matrix):
            dsc2 = elem.reshape((self.vector_size, 128))
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(dsc1, dsc2, k=2)
            score = 0
            # Apply ratio test
            for m, n in matches:
                if m.distance < st * n.distance:
                    score = score + 1
            result[index] = score
        return result

    def check_db(self, db_name="defaultdb.pck"):
        """
            Check before starting that the db with features is available.
            param:
                    db name: db to look at
          """
        # Try to create the DB if it doesn't exists.
        if not os.path.isfile(CFG_DIR + db_name):
            self.create_db(CFG_DIR + db_name)

    def match_img_db(self, im, db_name="defaultdb.pck"):
        """
            Query to db matcher
            param:
                    im: image to match
                    db_name: selected db
          """
        # Apply the same sift applied on DB and then load the db data.
        support = np.asarray(self.matrix, dtype=np.double)
        if support.size == 0:
            with open(CFG_DIR + db_name, "rb") as fp:
                self.data = pickle.load(fp)
            for k, v in self.data.items():
                self.names.append(k)
                self.matrix.append(v)
            self.matrix = np.array(self.matrix, dtype=np.float32)
            self.names = np.array(self.names)
        dsc = self.get_fv(im)
        if dsc is None:
            return None, None
        img_distances = self.dist(dsc, st=0.4)
        nearest_ids = np.argsort(img_distances)[::-1].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()
