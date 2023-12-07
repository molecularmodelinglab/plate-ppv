This is the code used to run the experiment in \<LINK TO PAPER GOES HERE\>

the curated dataset can be found in the `data` folder. You can also find these dataset on
[PubChem](https://pubchem.ncbi.nlm.nih.gov/) by searching the aid numbers

the code to generate the data collected in table one can be found in `main.py` and can be run with
`python main.py` (assumes `data/` directory is in CWD)

the code to generate figure 1 can be found in `plot_fig1.py` and require the results from `main.py`
(`results_xgboost_v2_bedroc.pkl`) and can be run with `python plot_fig1.py`

Python dependencies can be found in `requirements.txt`

`PlatePPV.py` holds a Sci-Kit Learn compatible PPV-of-the-top-N function with `N=128`. It requires probabilities, so you
can call it similar to how you call the `roc-auc-score` function in sklearn.
To use it, just copy the contents to your code. You can change `N` to any number you desire You can pass this 
function anywhere you can pass a Scorer object in sklearn, as it implements `make_scorer`