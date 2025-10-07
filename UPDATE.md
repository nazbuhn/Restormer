# Updated code implementation for Comparison of deep learning approaches for extreme low-SNR image restoration implementation 

<h1>Changes by file</h1>
<h2>Denoising/Options/RealDenoising_Restormer.yml</h2>
<ul>    
    <li>num_gpu set to 1.</li>
    <li>in_ch set to 1.</li>
    <li>inp_channels and out_channels set to 1.</li>
    <li>max_minibatch set to 64.</li>
</ul>
<h2>Denoising/generate_patches_sidd.py</h2>
<ul>
<li>Updated to take .tif files.</li>
<li>Splits files into training, validation, and test sets.</li>
<li>Removed patching and overlap logic because we use 512 x 512 non-overlapping crops.</li>
</ul>

<h2>Denoising/test_restormer.py</h2>
<h3>Updated the testing logic and added PSNR and SSIM evaluation to the testing script.</h3>
<ul>
<li>Updated the testing script to take dataset path, name, and weights.</li>
<li>Modified testing to use .tif files instead of .mat.</li>
<li>Paired images are preprocessed using CSBDeep helpers (sample_patches_from_multiple_stacks, no_background_patches, and norm_percentiles).</li>
<li> Images are normalized and PSNR and SSIM are calculated using skimage's implementation.</li>
</ul>

<h2>basicsr/data/paired_image_dataset.py</h2>
<ul>
<li>Images are split for training and testing.</li> 
<li>Paired images are preprocessed using CSBdeep helpers (sample_patches_from_multiple_stacks, no_background_patches and norm_percentiles) and preexisting normalization is removed.</li>
</ul>

<h2>Denoising/utils.py</h2>
<ul>
<li>Updated to contain image normalization performed prior to PSNR and SSIM calculation. Changes are as described in Image Quality Metrics.
</li>
</ul>

<h1>How to run training and testing on our dataset</h1>
<ul>
    <li>Update the training and val directory paths in /Denoising/Options/RealDenoising_Restormer.yml.</li>
    <li>Run: python3 -m train -opt /code/nbuhn/nbuhn/Restormer/Denoising/Options/RealDenoising_Restormer.yml. </li>
    <li>After training, cd into the Denoising folder and run: python3 -m test_restormer including --path, --dataset, and --weights.</li>
    
</ul>



