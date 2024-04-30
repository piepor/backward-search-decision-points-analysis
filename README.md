# Backward-Search for Decision Points Analysis in Petri Nets

Code for the chapter "*Backward-Search Decision for Decision Points Analysis in Petri Nets*" of the thesis "*Machine Learing for Probabilistic and Attribute-Aware Process Mining*".
The required packages are in the "*requirements.txt*" file. To install with pip:

```
pip install requirements.txt
```

To train the classifiers, specify the name of the base Petri net which must be in the directory "models/petri_nets". The resulting classifiers are in the same directory under "classifiers", while the results are in the "results" directory.

The available models are:
- Helpdesk_NT02
- BPI_Challenge_2013_incidents_NT02
- Road_Traffic_Fine_Management_Process
- Road_Traffic_Fine_Management_Process_NT02

where NT02 identifies the noise threshold of the inductive miner used for the discovery.
For example, to run the algorithm on "*Road_Traffic_Fine_Management_Process_NT02*":
```
python main.py Road_Traffic_Fine_Management_Process_NT02
```
