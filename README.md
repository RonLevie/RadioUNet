# RadioUNet
RadioUNet is a highly efficient and very accurate method for estimating the propagation pathloss from a point x to all points y on the 2D plane, in realistic propagation environments characterized by the presence of buildings. RadioUNet generates pathloss estimations that are very close to estimations given by physical simulation, but much faster. 

For more information see the paper [RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks](https://arxiv.org/pdf/1911.09002.pdf).



## Usage Examples

Download and extract the [RadioMapSeer dataset](https://drive.google.com/file/d/1PTaPpLOKraVCRZU_Tzev4D5ZO32tpqMO/view?usp=sharing) to the folder of the Jupyter Notebooks.
For training without samples see [RadioWNet_c_DPM_Thr2.ipynb](/RadioWNet_c_DPM_Thr2.ipynb).
For training with measurements and perturbed city map see [RadioWNet_s_randSim_miss4build_Thr2.ipynb](/RadioWNet_s_randSim_miss4build_Thr2.ipynb).
For training with simulated cars, measurements, and input car locations, see [RadioWNet_s_DPMcars_carInput_Thr2.ipynb](/RadioWNet_s_DPMcars_carInput_Thr2.ipynb).



