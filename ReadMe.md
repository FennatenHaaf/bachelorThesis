Welcome to my bachelor thesis code. This code implements methods to 
replicate p2v-map methodology from Gabel et. al (2019). It also implements
the t-sne method by van der Maaten & Hinton (2008)

The following is a description of each code file:

simulationExperiment.py contains the code used to create our simulated data, 
both with varying customer parameters and without.

createInstacartData.py contains  methods to process raw data from the Instacart dataset,  
merging various files so everything is available in one dataframe. Additionally, it provides 
methods to subsample the full dataset so that it is easier to use for training embeddings.

dataPrep.py contains classes that will prepare input data for being trained with the 
modified Skip-Grammodel. It removes infrequent products and creates pairs of center products 
with positive samples and 20 negative training samples.

productEmbed.py implements the modified Skip-Gram model.

productMap.py implements t-SNE to create a product map out of the embeddding outputs 
from productEmbed.py or out of pooled customer embeddings. It gives an option to create a 
product map that shows in theconsole, or an interactive one that is labeled with 
product ids (or in the case of the instacart data, names and department names).

benchmarkAndPool.py implements ways to plot loss, get average category similarity and 
co-occurences, calculate benchmarks and to get a random benchmark for embedding outputs from 
productEmbed.py. Additionally, it provides a method to pool product vector embeddings and 
show them in a customer map using the methods fromproductMap.py.

visualise.py contains methods to visualise some of the datasets, 
such as visualising correlations in aheatmap or visualising the counts in the Instacart dataset.

utils.py contains a few helper methods, like saving dataframes to csv and getting the current 
time.

BachelorThesisMain.py contains code to act as a central hub from which most of the methods in the 
otherfiles can be run