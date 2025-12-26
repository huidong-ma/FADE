<div align="center">
<h1>Efficient Learned Data Compression via Dual-Stream Feature Decoupling</h1>
</div>

# Usage
## Setup
```
cd ./FADE
pip install requirements.txt
```
## Running
```
# Compression
python sspp.py c <file> <file>.cmp

# Decompression
python sspp.py d <file>.cmp <file>.decmp
```
**For example**:
```
python sspp.py c enwik6 enwik6.cmp
python sspp.py d enwik6.cmp enwik6.decmp
```

# Dataset
| Dataset    | Type          | Description                                                                  | Link                                                           |
|:----------:|:-------------:|:----------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| Enwik9     | text          | First $10^9$ bytes of the English Wikipedia dump on 2006.                    | [Page](https://mattmahoney.net/dc/textdata.html)               |
| LJSpeech   | audio         | First 10,000 files of the LJSpeech audio dataset.                            | [Page](https://keithito.com/LJ-Speech-Dataset/)                |
| TestImages | image         | A classical 8-bit benchmark dataset for image compression evaluation.        | [Page](http://imagecompression.info/test_images/)              |
| UVG        | video         | The video ShakeNDry from the UVG benchmark featuring 1080p 8-bit YUV format. | [Page](https://ultravideo.fi/dataset.html)                     |
| CESM       | float         | First $10^9$ bytes of floating-point data from the CESM-ATM climate dataset. | [Page](https://sdrbench.github.io/)                            |
| DNACorpus  | genome        | A corpus of DNA sequences from 15 different species.                         | [Page](https://sweet.ua.pt/pratas/datasets/DNACorpus.zip)      |
| Silesia    | heterogeneous | A heterogeneous corpus of 12 files covering various file formats.            | [Page](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) |

The processed data used in the paper can be directly downloaded from [fade_datasets.tar.gz](https://drive.google.com/file/d/10UCnfr-WG-gevjl_n7nC0yP4A3sVaMPB/view?usp=drive_link) and extracted by executing `tar -xzf fade_datasets.tar.gz`.
