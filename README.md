# MoRE-Net

The official implementation of **MoRE-Net**, an interpretable model for MRI-based brain tumor grading that is robust to missing modalities.

## 🔧 Environment Setup

Please refer to [https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) for setting up the environment, including installation of the required dependencies:

- [`causal-conv1d`](https://github.com/ge-xing/SegMamba)
- [`mamba`](https://github.com/ge-xing/SegMamba)

Make sure these packages are properly installed before running the code.

## 🚀 Running the Code

To run the experiments, simply execute:

```bash
cd ./src
bash run.sh
```

Modify the configuration either:

- Directly in `run.sh`, or  
- Through the `parse_arguments()` function in `tumor_cls.py`

## 🙏 Acknowledgments

We would like to thank [https://github.com/aywi/mprotonet](https://github.com/aywi/mprotonet) for open-sourcing their code, which was helpful in the development of this project.

## 📄 Citation

If you find this work helpful, please consider citing the following paper:

```
XXX  # Replace this with your actual BibTeX citation
```
