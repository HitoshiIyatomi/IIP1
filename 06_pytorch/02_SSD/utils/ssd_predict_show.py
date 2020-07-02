"""
class SSDPredictShow: display the predicted results

"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCV library
import torch

from utils.ssd_model import DataTransform


class SSDPredictShow():
    """class for prediction SSD and displaying the results"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # class name
        self.net = net  # SSD network

        color_mean = (104, 117, 123)  
        input_size = 300  
        self.transform = DataTransform(input_size, color_mean)  # pre-process

    def show(self, image_file_path, data_confidence_level):
        """
        function for displaying results of SSD

        Parameters
        ----------
        image_file_path:  str
            file path
        data_confidence_level: float
           detection threshold for confidence

        Returns
        -------
        none
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        function for prediction with SSD

        Parameters
        ----------
        image_file_path:  strt
            image file path

        dataconfidence_level: float
            detection threshold for confidence

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # get rgb image
        img = cv2.imread(image_file_path)  
        height, width, channels = img.shape 
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pre-process
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # no annotation in prediction mode
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # prediction with SSD
        self.net.eval()  # set evaluation mode
        x = img.unsqueeze(0)  # mini-batch=1：torch.Size([1, 3, 300, 300])

        detections = self.net(x)
        # detections: torch.Size([1, 21, 200, 5]) 

        # detects box having larger confident than confidence_level
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # exec the loop for the number of objects extracted
            if (find_index[1][i]) > 0:  # not background
                sc = detections[i][0]  # confidence score
                bbox = detections[i][1:] * [width, height, width, height]
                # find_index: (# mini-batch, # class)
                lable_ind = find_index[1][i]-1 # because of background=0

                # append to return lists
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        function for displaying results of SSD

        Parameters
        ----------
        rgb_img:rgb image
            target image
        bbox: list
            list of object BBox
        label_index: list
            index of object label
        scores: list
            confidence score 
        label_names: list
            list of label name

        Returns
        -------
        none
        """

        # set frame color
        num_classes = len(label_names)  # num classes (exclude background)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # display image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox loop 
        for i, bb in enumerate(bbox):

            # label name
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # assign color for each class

            # label added to the frame e.g.:person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # coordinates of the frame
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # draw rectangle
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # draw label at the top-left of the rectanble
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})
