These scripts automate formation of the coco data sets necessary to train (many) object detection algorithms (such as yolact, YOLOv5 etc.). To run:

1 - pip install -r requirements
Would recommend a virtual environment

2 - Create a classDict.json file in the format:
{"<label>": <"id">}
For each data source to specify which data sources you want to use from each. It is RECOMMENDED that you create one classDict.json and copy them into all data sources. 

Where id < 0 and the values are preferablly sequential.
Identifying the different data types is automatic so if for example you had datasourece "a", "b" and "c" but only created the classDict.json file:
{"a": 1, "b": 2}, then only the labels a and b in all data sources will be accessed for the coco data structure format.

3 - Run createSourceInfo.py. This creates the individuaal annotation, image and categories information.

4 - Run createCocoDataSet.py. This combines all the individual annotation files and then splits them into training, validation and testing coco formats.

5 (optional) - Run overlaySegment.py to assess if the annotations have been correctly created for the data sets.