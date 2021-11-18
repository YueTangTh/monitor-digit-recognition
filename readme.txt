
========================= code file list =============================

	1. v.py [main program]
		goal: 	recognize the digits and output signals
		input: 	cropped videos in <test> folder
		output: images, datasheets(.npy & .mat) in <result> folder
		
	2. train_model.py
		goal: 	train the digit recognition model
		input: 	training data in <training> folder
		output: knn network in <knn> folder
	
	3. seg.py
		goal:	crop the videos - focus the window on digits
		input:	long videos (25mins - 1h)
		output:	cropped vidoes
	
	4. plot.py
		goal:	plot the signals of each parameter
		input:	.npy data from <result/data> folder
		output:	signal graphs
		
		
=========================== folder list =============================

	1. DisplayUtils:	display tools - not very useful
	2. ImageProcessing:	image processing module of the network
						especially - FrameProcessor.py
	3. knn:	store the knn network
	4. result: result folder, datasheets, images
	5. test: test_folder, cropped videos
	6. training: training data for training the network
