# RadioUNet
RadioUNet is a highly efficient and very accurate method for estimating the propagation pathloss from a point x to all points y on the 2D plane, in realistic propagation environments characterized by the presence of buildings. RadioUNet generates pathloss estimations that are very close to estimations given by physical simulation, but much faster. 

For more information see "RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks."



## Usage

Download and extract the [RadioMapSeer dataset](https://drive.google.com/open?id=1Lqxf2b1vL41BW4-PLXutDv1I4Z2UNgtV).
For training without samples see [RadioUNet_c.ipynb](/RadioUNet_c.ipynb), and for training with measurements and perturbed urban geometry see [RadioUNet_s.ipynb](/RadioUNet_s.ipynb).
