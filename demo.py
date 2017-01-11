from face_frontalization import frontalize, facial_feature_detector as feature_detection, camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

this_path = os.path.dirname(os.path.abspath(__file__))


def demo():
    model3D = frontalize.ThreeD_Model.default_model()
    # load query image
    img = cv2.imread("test.jpg", 1)
    plt.title('Query Image')
    plt.imshow(img[:, :, ::-1])
    # extract landmarks from the query image
    # list containing a 2D array with points (x, y) for each face detected in the query image
    lmarks = feature_detection.get_landmarks(img)
    plt.figure()
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])
    # perform camera calibration according to the first face detected
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    # load mask to exclude eyes from symmetry
    eyemask = frontalize.EyeMask.get_default()
    # perform frontalization
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    plt.figure()
    plt.title('Frontalized no symmetry')
    plt.imshow(frontal_raw[:, :, ::-1])
    plt.figure()
    plt.title('Frontalized with soft symmetry')
    plt.imshow(frontal_sym[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    demo()
