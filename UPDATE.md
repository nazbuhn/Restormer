# Updated code implementation for Comparison of deep learning approaches for extreme low-SNR image restoration implementation 

<h1>Changes made per file</h1>
<h2>Denoising/Options/RealDenoising_Restormer.yml</h2>
<ul>    
    <li>num_gpu set to 1</li>
    <li>in_ch set to 1 for Greyscale images for train and test set</li>
    <li>inp_channels and out_channels: set to 1</li>
    <li>max_minibatch set to 64</li>
</ul>
<h2>Denoising/generate_patches_sidd.py</h2>
<ul>
<li>Changed to take .tif files from .png</li>
<li>Split files into train, test and validation sets</li>
<li>Remove patching and overlap logic as we use 512 x 512 nonoverlapping crops</li>
</ul>

<h2>Denoising/test_restormer.py</h2>
<h3>Updeated testing logic and include PSNR and SSIM evaluation in testing script.</h3>
<ul>
<li>Testing script takes dataset path and name, along with weights</li>
<li>Changed testing to take .tif files instead of .mat.</li>
<li>Paired images are preprocessed using CSBdeep helpers (sample_patches_from_multiple_stacks, no_background_patches and norm_percentiles)</li>
<li> Images were normalized as described in (Supplementary Notes, Section 2.2 of Weigert et al. (2018)) and PSNR and SSIM were calculated skimage's implementation. </li>
<li>Handles single-channel images writes as float32 .tif files</li>
</ul>

<h2>basicsr/data/paired_image_dataset.py</h2>
<ul>
<li>Images are split for training and testing</li> 
<li>Paired images are preprocessed using CSBdeep helpers (sample_patches_from_multiple_stacks, no_background_patches and norm_percentiles) and prexisiting normalization is removed </li>
</ul>

<h2>Denoising/utils.py</h2>
<ul>
<li>Changed utils to contain image normalization done prior to PSNR and SSIM calculation. Changes are as referenced in Image Quality Metrics Section following(Supplementary Notes, Section 2.2 of Weigert et al. (2018).)
</li>
</ul>

<h1>How to run training and testing on our dataset</h1>
<ul>
    <li>Update path to raw and gt directories in /Denoising/Options/RealDenoising_Restormer.yml for train and val set</li>
    <li>Run python3 -m train -opt /code/nbuhn/nbuhn/Restormer/Denoising/Options/RealDenoising_Restormer.yml </li>
    <li>After training cd into Denoising folder and run python3 -m test_restormer with --path, --dataset, and --weights</li>
    
</ul>



