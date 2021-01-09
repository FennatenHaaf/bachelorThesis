# -*- coding: utf-8 -*-
"""
Implements methods to represent embedding vectors in a two-dimensional map,
using t-SNE (van Maaten 2014)

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import numpy as np
import pandas as pd
import sklearn.manifold
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import seaborn as sns
sns.set_style("whitegrid")

# My own modules
import utils
import visualise



class product_mapping:
    """Contains methods to make a product mep out of vectors trained with the
    product2vec method"""
    
    def __init__(self, base_name, data_dir, embedding_dir,
                 data_type, epoch, batch, otherdir = None, seed = 1234, 
                 interactive = True, norm = True ):
        
        print(f"processing {data_type} vectors for tSNE plotting,"
              f" at {utils.get_time()} ")
        
        #-------------------------INITIALISATION-----------------------
        
        self.base_name = base_name # the string forming the base for most 
        # of the data, e.g. "gabel_baskets"
        self.data_dir = data_dir # the directory containing the data
        self.embedding_dir = embedding_dir # directory containing the models
        self.data_type = data_type # can be "instacart", "simulated" or "gabel"
        self.epoch = (epoch - 1) # The epoch num that the embedding file has
        self.batch = batch # The batch num that the embedding file has
        self.otherdir = otherdir # a non-standard location of the embeddding files
        self.seed = seed # random seed for PCA decomposition / tsne
        self.interactive = interactive
        self.norm = norm # Boolean for if we should apply L2 normalisation,
        # which should usually be done for L>10
 
        assert (self.data_type in {"instacart", "simulated", "gabel","simulated_c2v"}), \
            "valid data types are: instacart, simulated, gabel or simulated_c2v"
        
        #-------------------------LOAD EMBEDDINGS-----------------------
        # model = load_model(f"{self.indir}/{self.model_name}")
        # embedding_layer = model.get_layer('center_embedding')
        # embeddings = np.array(embedding_layer.get_weights()[0])
        
        print(f"loading embedding file wi_{self.epoch}_{self.batch} from"
              f" {self.embedding_dir}/{self.data_type}")
        
        if otherdir is None:
            embeddings = np.load(f'{self.embedding_dir}/{self.data_type}/wi_{self.epoch}_{self.batch}.npy')
        else: # if the embeddings are located in a non-standard location
            embeddings = np.load(f'{self.embedding_dir}/{self.otherdir}/wi_{self.epoch}_{self.batch}.npy')
        
        #-------------------------L2 NORMALISATION-----------------------
        if norm:
            print("doing L2 normalisation")
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            # print(pd.DataFrame(embeddings).head())
            
        #-------------------------GET TSNE EMBED---------------------------
        #TSNE step by van Maaten (2014)
        
        print("getting tsne output")
        
        tsne_output = sklearn.manifold.TSNE(random_state=seed, # a seed num
                                        n_components=2, # barnes hut dimension
                                        n_iter=4000, # max iter for optimization
                                        init = 'pca',
                                        perplexity=15, # used in barnes hut
                                        angle=0.5, # used in barnes hut
                                        verbose=1).fit_transform(embeddings)
        
        # https://distill.pub/2016/misread-tsne/ - a nice link to show how
        # tsne can change based on different parameters
        

        #----------------------------PLOT DATA---------------------------
        
        tsne_map_xy = self.get_plot_df(tsne_output)

        self.num_prod= len(tsne_map_xy)
   
        print(f"Plotting map with {self.num_prod} products at {utils.get_time()}")
        if self.interactive:
            plot_map_interactive(tsne_map_xy,self.data_type)
        else:
            plot_product_map(tsne_map_xy,self.data_type)


        
    def get_plot_df(self,tsne_output,cat_name = "c", prod_id = "j"):
        """The goal of this function: to link the x and y values of the
        embeddings created, with the product and category names 
        """   
        indir = self.data_dir
        product_vec =  pd.read_csv(f"{indir}/{self.base_name}_center_products_train.csv")
        
        if self.data_type == "gabel":
             data = pd.read_csv(f"{indir}/{self.base_name}_train_final.csv")
        else:
            data = pd.read_csv(f"{indir}/{self.base_name}_split_train.csv")
                    
        ## Now put the relevant data into a dictionary
        
        tsne_data = {}
        tsne_data.setdefault("c", []) #category
        tsne_data.setdefault("j", []) #product
        tsne_data.setdefault("x", []) #x coordinate
        tsne_data.setdefault("y", []) #y coordinate
        
        if self.data_type == "instacart":
            tsne_data.setdefault("cat", []) #category name
            tsne_data.setdefault("dept", []) #add department name as well
            tsne_data.setdefault("dept_id", []) #add department name as well
        
        for i,product in enumerate(product_vec.iloc[:,0].unique()):
            product = int(product)
            
            x = tsne_output[product,0] 
            y = tsne_output[product,1]
            
            # Get the category corresponding to this product
            category = data[cat_name][data[prod_id]==product].unique()
            category = int(category[0])
            
            if self.data_type == "instacart": # for instacart we can use names
                product_name = data["prod_name"][data[prod_id]==product].unique()
                product_name = product_name[0]
                category_name = data["aisle"][data[prod_id]==product].unique()
                category_name = category_name[0]
                department_name = data["department"][data[prod_id]==product].unique()
                department_name = department_name[0]
                department_id = data["department_id"][data[prod_id]==product].unique()
                department_id = department_id[0]
                
                tsne_data["cat"].append(category_name)
                tsne_data["j"].append(product_name)
                tsne_data["dept"].append(department_name)
                tsne_data["dept_id"].append(department_id)
            
            else: # for the others we have to use numbers for the prod name
                tsne_data["j"].append(product)
            
            # These are the same for all three
            tsne_data["x"].append(x)
            tsne_data["y"].append(y)
            tsne_data["c"].append(category)            
            
            
        return pd.DataFrame(tsne_data)



def plot_product_map(data, data_type, cust_map = False):
    """Plot a product map to show in the plots console"""

    tsne_data = data.reset_index()
    sns.set(font_scale=2,rc={'figure.figsize':(20,15)})
    #sns.set_style("whitegrid")

    if data_type == "instacart": 
        tsne_data["cat"] = tsne_data["cat"].astype('category')
        tsne_data["dept"] = tsne_data["dept"].astype('category')
        
        # can visualise hue by dept or by cat
        graph = sns.scatterplot(x = "x", y= "y", 
                                 data=tsne_data, hue="dept", s=60) 
        
    elif data_type == "c2v": # don't give it a particular hue
        graph = sns.scatterplot(x = "x", y= "y", data=tsne_data, s=60)
        
    elif data_type == "simulated_c2v_pooled": 
        tsne_data["cat_i"] = tsne_data["cat_i"].astype('category')
        graph = sns.scatterplot(x = "x", y= "y", data=tsne_data,
                                hue = "cat_i", s=60)
    else:
        tsne_data["c"] = tsne_data["c"].astype('category')
        graph = sns.scatterplot(x = "x", y= "y", 
                                 data=tsne_data, hue="c", s=60)

            
    # move legend to the upper right next to the figure
    # - see https://matplotlib.org/3.1.1/tutorials/intermediate/legend_guide.html
    graph.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1) #ncol=2 for larger things
    plt.show()
    
        
        
def plot_map_interactive(data,data_type):
    """
    This is to plot an interactive product map, from Gabel et al (2019)
    """
    dt = data.reset_index()
    filename = f"{data_type}.html"

    if data_type == "instacart":
        plot_data = [
            go.Scatter(
                x=dt['x'].values,
                y=dt['y'].values,
                text=[
                    f'department={c}<br>category={x}<br>product={y}'
                    for (c, x, y) in zip(dt['dept'].values,
                                        dt['cat'].values, dt['j'].values)
                    ],
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=14,
                    color=dt['dept_id'].values, # make the departments determine colour
                    colorscale='rainbow', # https://plotly.com/python/builtin-colorscales/
                    showscale=False
                )
            )
        ]
    elif data_type == "simulated_c2v_pooled":
        plot_data = [
            go.Scatter(
                x=dt['x'].values,
                y=dt['y'].values,
                text=[
                    f'id={i}<br>cust_category={c_i}'
                    for (i, c_i) in zip(dt['i'].values,
                                        dt['cat_i'].values)
                    ],
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=14,
                    color=dt['cat_i'].values, # make the categories determine colour
                    colorscale='rainbow', # https://plotly.com/python/builtin-colorscales/
                    showscale=False
                )
            )
        ]
        
        
    else:
         plot_data = [
            go.Scatter(
                x=dt['x'].values,
                y=dt['y'].values,
                text=[
                    f'category = {x}<br>product = {y}'
                    for (x, y) in zip(dt['c'].values, dt['j'].values)
                    ],
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=14,
                    color=dt['c'].values, # make the categories determine colour
                    colorscale='rainbow', # https://plotly.com/python/builtin-colorscales/
                    showscale=False
                )
            )
        ]

    plot_layout = go.Layout(
        width=800,
        height=600,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
        hovermode='closest'
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(fig, filename=filename)
    
    #print(f"Done, at {utils.get_time()}")
    
    
    
if __name__ == '__main__':

    simulation = False
    instacart = False
    gabel = True
    
    # if simulation:
    #     print("Doing tsne for simulation data")
    #     data_indir = "./largeData"
    #     model_indir = "./models"
    #     model = "simulated_t100_c1000_epoch8_L30.h5" 
        
    #     simulation_map = product_mapping(model_indir, model, data_type = "simulation",
    #                                      seed=1)
        
    # if instacart:
    #     print("Doing tsne for instacart data")
    #     data_indir = "./instacart_2017_05_01"
    #     model_indir = "./models"
    #     model = "instacart_cat40prod10seed1234_epoch5_L15_2.h5"
        
    #     instacart_map = product_mapping(model_indir, model, data_type = "instacart",
    #                                     seed=1)
    
    # if gabel:
    #     print("Doing tsne for gabel data")
    #     data_indir = "./largeData"
    #     model_indir = "./models"
    #     #model = "original_baskets_epoch5_L15.h5" #Degene die lang is getraind
    #     #output_indir = "./results/p2v-map-example"
    #     output_indir = "./output"
        
    #     instacart_map = product_mapping(output_indir, data_type = "gabel",
    #                                     epoch = 4, batch = 4000,
    #                                     seed=1)
        
    
                                          