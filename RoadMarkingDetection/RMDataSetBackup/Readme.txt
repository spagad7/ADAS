Roadmarking dataset
-------------------

The dataset consists of 1443 images of road markings and groundtruth position
and labels are given in the file dataset_annotations.txt in CSV format as
follows -

x1,y1,x2,y2,x3,y3,x4,y4,label,image_filename

where the first 8 numbers denote the bounding box in the image in pixel 
coordinates.
Questions about the dataset can be sent to 
Ananth Ranganathan (aranganathan@honda-ri.com) or Tao Wu (taowu@umiacs.umd.edu)

Please cite the following paper if using this dataset -

"A Practical System for Road Marking Detection and Recognition",
Tao Wu and Ananth Ranganathan
IEEE Intelligent Vehicles Symposium, 2012.

Calibration information is as follows -
The transform matrix A -
[0.197907   -0.178787       360.82  ]
[0          -0.00504166     50.9823 ]
[0          -0.000446185    1       ]

And the POI boundaries are:
x: [-1000, 1500] and y: [-1200, 1800]
where the transform matrix A is defined for the transform equation: [u,v,f]' = A' [x-250, y, 1]'
[u/f, v/f] are the coordinates of the pixels in transformed image. [x,y] are the coordinates of the pixels in the original image. The prime ' means matrix transpose. The x-250 part is a simple crop on the original image so that the sky part is omitted. This can be used to reproduce the results in the paper.

The array bound is a reference boundary of the ROI in the transformed coordinates.

More details can be found in the paper.