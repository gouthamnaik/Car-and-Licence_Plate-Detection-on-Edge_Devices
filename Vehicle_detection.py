import cv2
import numpy as np

import colorsys
import glob
import os

import tensorflow as tf
        # Using tf_lite to make it work on Edge_devices
from tflite_runtime.interpreter import Interpreter


def filter_boxes(box_xywh, scores, score_threshold = 0.4, input_shape = tf.constant([416, 416])):
    scores_max = tf.math.reduce_max(scores, axis = -1)
    mask = scores_max >= score_threshold
    
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
    
    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
    input_shape = tf.cast(input_shape, dtype=tf.float32)
    
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape

    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return (boxes, pred_conf)
    

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
    

def draw_bbox(image, bboxes, classes = read_class_names("data/classes/classes.names"), show_label = True):
            # Number of classes
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    detected_classes = [classes[int(out_classes[0][i])] for i in range(num_boxes[0])]
    print("Found this classes...")
    print(detected_classes)
    
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
       
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        
        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]
                # Rect_box color
        bbox_color = colors[class_ind]
                # Rect_box thickness
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
                # Drawing rectangle for the detected objects in the img
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
                    # Giving name to the label
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)                            
    return image


def main():
            # Size of an img
    input_size = 416
            # Output path, you can give your own path
    output = "data/images/output"
            # My trained model path
    model_path = "model/vehicle-tiny-416.tflite"
            # Threshold
    iou = 0.45
    score = 0.25
    
            # Assigning my trained model to the tf_lite interpreter
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
            # Accesing imgs to be detected
            # For single img, remove (*) and give your own img(jpg/png/jpeg)
            # But here I am giving the comp img folder(*), which will detect every img in that folder
    folder = glob.glob("data/images/input/*")
    
    for image_path in folder:
        print("Reading this img...")
        print(image_path)
                # This gives me only img name not complete path of the img
        image_name = os.path.basename(image_path)
                # Giving img path to the variable
        original_image = cv2.imread(image_path)
                # Changing color BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Resizing the img to (416, 416)
        image_resize = cv2.resize(original_image, (input_size, input_size))
        image_data = np.expand_dims(image_resize, axis = 0)
        image_data = (np.float32(image_data) / 255.)
        
                # Perform the actual detection by running the model with the img as input
        interpreter.set_tensor(input_details[0]["index"], image_data)
        interpreter.invoke()
        
        pred = [interpreter.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]
        
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold = 0.25, input_shape = tf.constant([input_size, input_size]))
        
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes = tf.reshape(boxes, 
            (tf.shape(boxes)[0], -1, 1, 4)), scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, 
            tf.shape(pred_conf)[-1])), max_output_size_per_class = 50, max_total_size = 50, iou_threshold = iou,
            score_threshold = score)
    
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        image = draw_bbox(original_image, pred_bbox)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        output_path = output + "/" + image_name
        print("Saving at...")
        print(output_path)
        print("-" * 50)
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    main()
