
This repository contains some demos for the [HGQ](https://github.com/calad0i/HGQ) library.

To prepare the environment, install the required packages by running:

```bash
pip install -r requirements.txt
```

To prepare the data, download the datasets from the following links:

- Jet Classifier:
  - No need to download dataset manually; automatically fetching from openml.
- SVHN: http://ufldl.stanford.edu/housenumbers/
  - Place all three files (`train_32x32.mat`, `test_32x32.mat`, `extra_32x32.mat`) in the `data/svhn` directory.
- TGC: https://huggingface.co/datasets/Calad/fake-TGC/blob/main/fake_TGC_0.041_pruned.h5
  - Place the hdf5 file in the `data/tgc` directory.
- Jet Classifier Large:
  - Fetch https://zenodo.org/records/3602260, prepare the dataset with `jet_classifier_large/prepare_dataset.py -i <path_to_files> -o <output_path>` to `train.h5` and `test.h5`, and place them in some directory.
  - Update the `datapath` in the configuration files to point to the directory containing the `train.h5` and `test.h5` files.
  - Replace `model.py` with `model_*.py` to use the MLP and/or fp32 models. The default is quantized MLP-Mixer.

To execute the demos, `cd` into the individual demo directory run the corresponding python script. For example, to run the demo for the jet classifier, execute the following commands:

```bash
cd jet_classifier
python jet_classifier.py -c configs/<config_file>.json -r all
```

For more details, please run the script with the `-h` flag.
You may want to modify the output paths in the configuration files.

For synthsizing the models, you can use the `batch_synth.sh` script provided, which runs csynth, vsynth, and place&route, and keeps only the reports. Copy the script to the directory containing the tarballs of the hls projects, and run it.

If you wish to do it manually, please notice the following:

- When using dense-only network with `vivado_hls` with function pipeline, add `#pragma HLS INLINE recursive` in the entry function. This will greatly reduce the latency for HGQ trained models.
- For the tgc Muon tracking demo, replace `#pragma HLS DATAFLOW` by `#pragma HLS PIPELINE`. Vitis/Vivado HLS does a bad job with dataflow for this model.
  - Probably due to the fill buffer logic in for convolutional layers. 

An ipython notebook containing the minimal code to run the HGQ is available [here](minimal/usage_example.ipynb)
