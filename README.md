# map-equation-centrality
This repository contains companion notebooks for _Map Equation Centrality: Community-aware Centrality with the Map Equation_ for reproducibility.

## Setup
To get started, create a virtual environment `virtualenv map-equation-centrality-venv` and activate it `source map-equation-centrality-venv/bin/activate`, preferably using python 3.9.7 compiled with GCC 11.2.0.
Then, install the required packages with `pip install -r requirements.txt`.
Start a jupyter server and run the notebooks you are interested in.

## Data
The datasets that are used can all be found on [Netzschleuder](https://networks.skewed.de/), and each of the notebooks download the data that it uses.

We use the following data sets:
- [facebook-friends](https://networks.skewed.de/net/facebook_friends)
- [copenhagen/fb_friends](https://networks.skewed.de/net/copenhagen)
- [uni-email](https://networks.skewed.de/net/uni_email)
- [polblogs](https://networks.skewed.de/net/polblogs)
- [interactome-yeast](https://networks.skewed.de/net/interactome_yeast)
- [ego-facebook/facebook_combined](https://networks.skewed.de/net/ego_social)
- [power](https://networks.skewed.de/net/power)
- [facebook-organizations/L2](https://networks.skewed.de/net/facebook_organizations)
- [physics-collab/arXiv](https://networks.skewed.de/net/physics_collab)
- [google](https://networks.skewed.de/net/google)
- [pgp](https://networks.skewed.de/net/pgp_strong)
- [facebook-wall](https://networks.skewed.de/net/facebook_wall)

## SIR disease spreding power
The notebooks include evaluations that use the SIR spreading power of the nodes, that is, the expected number of nodes that will be infected in an SIR epidemic when only the respective node is the initial spreader (see the manuscript for more details).
This repository includes the spreading powers for the used networks, but if you want to re-calculate them yourself, you will need [this tool](https://github.com/chrisbloecker/spreading-power).
