def get_class(image, classes, category_index,):
	# if not max_boxes_to_draw:
	# 	max_boxes_to_draw = boxes.shape[0]
	for i in range(boxes.shape[0]):
		if scores is None or scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())
			if classes[i] in category_index.keys():
					class_name = category_index[classes[i]]['name']
					if class_name == 'person':
						print(image)
					else:
						class_name = 'N/A'
