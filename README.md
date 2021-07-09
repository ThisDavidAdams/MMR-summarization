# MMR-summarization
This is the repository for our research involving the combination of state-of-the-art models using MMR.

# Requirements
* The libraries in the requirements.txt file.
* Mallet, which should be downloaded to a mallet-2.0.8 folder accessible to the code (recommended C:\mallet-2.0.8 for offline use, or the default location prior to mounting a drive for Colab), and a MALLET_HOME path variable should be set on your system.
* Spacy. Use "python3 -m spacy download en_core_web_sm" to download spacy for loading in the LDA computation.
* The MatchSum Multi-News output in JSONL format, located in the matchsum_output folder. The DEC files for the output can be found in the MatchSum repository [here](https://github.com/maszhongming/MatchSum), in the Generated Sumamries section of the README file. We have provided a simple script for converting multiple DEC files into a single JSONL file in the matchsum_output folder. Simply put the MatchSum authors' multinews_output folder in our matchsum_output folder to use our convert_dec_to_jsonl.py script with the default --input and --output arguments.

# To Use
There are two experiments included in this repository:
* The Multi-News experiment, which uses the Multi-News dataset.
  * mmr_combination_multinews.ipynb is a notebook which contains instructions for running the code in colab, assuming you have the source files in your Google Drive in a MMRSumm folder.
  * mmr_combination_multinews.py is the source code for the experiment.
  * optimize_mmr_combination_multinews.py is the script for optimizing the experiment.
  * mmr_filter.py is used by both other scripts, and contains the code to perform the MMR calculations.

* The WCEP experiment, which uses the WCEP dataset.
  * WCEP_Dataset.ipynb is a notebook containing code which prepares the WCEP dataset in your Google Drive MMRSumm folder.
  * WCEP_MMR_Summarisation.ipynb is a notebook containing code which sumnmarizes, optimizes, and analyzes the performance of MMR-combined summarization of the WCEP dataset.
