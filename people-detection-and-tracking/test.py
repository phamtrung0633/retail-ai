import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
model_predictions  = yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg", conf=0.5)

prediction = model_predictions[0].prediction # One prediction per image - Here we work with 1 image, so we get the first.

bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object
print("Poses: ", len(prediction.poses[0]))