# -*- coding: utf-8 -*-
"""
Implements several different methods to evaluate the output of our p2v-map
implementation. Also implements a way to pool the output center embeddings into
customer embeddings, and shows it in a product map.

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import numpy as np                  
import pandas as pd
from tqdm import tqdm
import seaborn as sns
sns.set_style("whitegrid")
import sklearn.cluster
import sklearn.manifold
import scipy.spatial.distance
import matplotlib.pyplot as plt


# My own modules
import utils
import productMap
import visualise


class evaluationClass:
    
    def __init__(self, base_name, data_dir, embedding_dir,
                  specific_dir,
                 data_type, max_epoch, max_batch, step_size =1000):
        """Initialisation for evaluationClass"""
         
        print(f"Evaluating {data_type} embedding results,"
                f" at {utils.get_time()} ")
        
        #-------------------------INITIALISATION-----------------------
        
        self.base_name = base_name # the string forming the base for most 
        # of the data, e.g. "gabel_baskets"
        self.data_dir = data_dir # the directory containing the data
        self.embedding_dir = embedding_dir # directory containing the models
        self.data_type = data_type # can be "instacart", "simulated" or "gabel"
        self.specific_dir = specific_dir # The specific directory containing our files
        self.max_epoch = max_epoch # The max epoch num that the embedding file has
        self.max_batch = max_batch # The max batch num that the embedding file has
        self.step_size = step_size # The size of the batches, used for getting files 
     
        
        assert (self.data_type in {"instacart", "simulated", "gabel","simulated_c2v"}), \
            "valid data types are: instacart, simulated, gabel or simulated_c2v"
               
        #------------------------LOAD FILES-----------------------
        
        self.files_per_epoch =(max_batch // step_size) # number of files per epoch
        
        # Get the full dataset for product info
        if self.data_type == "gabel":
            self.full_data = pd.read_csv(f"{self.data_dir}/{self.base_name}_train_final.csv")
        else:
            self.full_data = pd.read_csv(f"{self.data_dir}/{self.base_name}_split_train.csv")
         
        # Get the unique products that were used for embeddings
        self.product_vec = pd.read_csv(f"{self.data_dir}/{self.base_name}_"
                                   "center_products_train.csv")
        self.product_vec = sorted(self.product_vec.iloc[:,0].unique())
        
        # Get the unique categories used
        self.unique_categories = self.full_data["c"].unique()
        
        # Load loss and load embedding data
        self.loss_dict = self.load_all_files("train_loss")
        self.wi_embeddings = self.load_embeddings("wi")
        self.wo_embeddings = self.load_embeddings("wo")
        
        # Now get dataframe with x and y coordinates
        self.wi_xy = pd.DataFrame(self.get_xy_dict(self.wi_embeddings))
      
        #----------------------GET BENCHMARKS-----------------------
      
        print(f"Calculating benchmarks for {self.data_type} data")
        ss, amis, nnh = self.calculate_benchmarks(self.wi_xy)
      
        print(f"Done calculating benchmarks at {utils.get_time()}! \n"
              f" Silhouette score: {ss} \n"
              f" Adjusted mutual info score:{amis} \n"
              f" nn hitrate: {nnh}")
        
      
        
        
    def load_embeddings(self,base_string):
        """Load the embedding weights and normalise"""
        
        embeddings = np.load(f"{self.embedding_dir}/{self.specific_dir}/"
                              f"{base_string}_{(self.max_epoch-1)}_"
                              f"{self.max_batch}.npy")
        # Doing L2 normalisation
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
        return embeddings
  
    
    def get_xy_dict(self, embeddings, n=1, seed=1):
        """Get a dataframe of x,y coordinates using tsne"""
        
        print(f"Running tsne output {n} times and selecting result with"
              " lowest divergence") 
        
        # Kullback-Leibler divergences of t-SNE solutions are directly
        # comparable between runs when perplexity remains fixed, so we can
        # run it 10 times and choose the output with lowest divergence
        # - see author github faq: https://lvdmaaten.github.io/tsne/ 
        best_kl = float('inf')
        best_so_far = None
        
        for i in range(n): #set to 10 for testing
            model = sklearn.manifold.TSNE(random_state=seed, # a seed num
                                    n_components=2, # barnes hut dimension
                                    n_iter=4000, # max iter for optimization
                                    init = 'pca',
                                    perplexity=15, # used in barnes hut
                                    angle=0.5, # used in barnes hut
                                    verbose=0)      
            current_output = model.fit_transform(embeddings)
            kl = model.kl_divergence_
            
            if (kl < best_kl): # The newest divergence is lower than the previous best
                best_so_far = current_output
                best_kl = kl
                
        print(f"Lowest divergence was {best_kl}")
                
        tsne_output = best_so_far
        
        ## Now put the output into a dictionary
        xy_dict = {}
        xy_dict.setdefault("c", []) #category
        xy_dict.setdefault("j", []) #product
        xy_dict.setdefault("x", []) #x coordinate
        xy_dict.setdefault("y", []) #y coordinate
        
        for j,product in enumerate(self.product_vec):
            product = int(product)
        
            x = tsne_output[product,0] 
            y = tsne_output[product,1]
            
            ## Get the category corresponding to this product
            category = self.full_data["c"][self.full_data["j"]==product].unique()
            category = int(category[0])
        
            ## Add the results
            xy_dict["x"].append(x)
            xy_dict["y"].append(y)
            xy_dict["c"].append(category)
            xy_dict["j"].append(product)
            
        return xy_dict
    
    

    def calculate_benchmarks(self, xy_df, pooled = False):
        """ This function calculates the common benchmarking metrics
        shillouette score, adjusted mutual info score and nn hitrate"""

        #-----------Silhouette & adjusted mutual info score-------------
      
        # First we need to get the 'true' clusters based on the 
        # actual categories
        if pooled: # in this case, we are calculating for pooled customer vecs
            xy_df = pd.DataFrame(xy_df).rename(columns = {"cat_i":"c",
                                                          "i":"j"})
            
        true_clusters = xy_df['c']
        num_clusters = len(true_clusters.unique())
        
        # Split into xy, j and c
        data_xy = xy_df[["x","y"]].to_numpy()
        j = xy_df["j"].to_numpy()
        c = xy_df["c"].to_numpy()
                
        xy_df = xy_df.set_index(['j', 'c'])
        
        ## Now cluster it
        clustering = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=30)
        kmean_pred = clustering.fit_predict(xy_df.values)
        
        # change to numpy for this
        xy_arr = xy_df.to_numpy()
        true_clusters = true_clusters.to_numpy()
        
        # get silhouette score & adjusted mutual info
        silhouette_score = sklearn.metrics.silhouette_score(X=xy_arr, 
                                                            labels=true_clusters)
        adjusted_info = sklearn.metrics.adjusted_mutual_info_score(
            labels_true=true_clusters,
            labels_pred=kmean_pred,
            average_method='arithmetic')
        
        #------------------------nn hitrate------------------------
      
        ## Now get the number of products per cluster:        
   
        # TODO: Make this more flexible rather than hardcoding?
        if pooled:
            J_C = 100
        elif self.data_type == "instacart":
            J_C = 10
        else:
            J_C = 15
        
        #For the hitrate, we need the number of nearest neighbors per cat 
        #which would be in the same cluster as some specific product, therefore
        #it should be J_C (prods per category) minus 1
        num_neighbours = (J_C-1)
              
        # gets euclidean distance
        distances = scipy.spatial.distance.cdist(data_xy, data_xy) 
        distance_df = pd.DataFrame({'j': np.repeat(j, len(j)), # get an array of j len(j) times
                                    'c': np.repeat(c, len(c)),
                                    'j2': np.tile(j, len(j)), # np.tile repeats an array
                                    'c2': np.tile(c, len(c)),
                                    'd': distances.flatten() })
        ## Make sure that the distance of products to themselves is taken out
        distance_df = distance_df[distance_df['j'] != distance_df['j2']]
        distance_df = distance_df.sort_values('d') # sort by distance
        
        ## get only the specified number of closest neighbours
        distance_df['rank_d'] = distance_df.groupby('j').cumcount()
        nn = distance_df[distance_df['rank_d'] < num_neighbours]
        
        hitrate = float(sum(nn['c'] == nn['c2'])) / nn.shape[0]

        
        return silhouette_score,adjusted_info,hitrate
    
    
    def get_random_benchmarks(self):
        """We make a product map of random coordinates, and use that to
        calculate benchmark scores"""
        
        print("Calculating random benchmarks")
        
        ## Make a dictionary with xy coordinates
        
        xy_dict = {}
        xy_dict.setdefault("c", []) #category
        xy_dict.setdefault("j", []) #product
        xy_dict.setdefault("x", []) #x coordinate
        xy_dict.setdefault("y", []) #y coordinate
        
        # We want it in the same size as our normal df so we iterate over product
        # number
        for j,product in enumerate(self.product_vec):
            product = int(product)
        
            # Get random coordinates within a unit circle, using a
            # polar coordinate system, y1 = r cos(α), y2 = r sin(α),
            # with r2 ∼ U(0, 1) and α ∼ 2 𝜋 U(0, 1) .
            r = np.random.uniform(0, 1)
            alpha = 2 * np.pi * np.random.uniform(0, 1)
    
            x = np.sqrt(r) * np.cos(alpha)
            y = np.sqrt(r) * np.sin(alpha)
            
            ## Get the category corresponding to this product
            category = self.full_data["c"][self.full_data["j"]==product].unique()
            category = int(category[0])

            ## Add the results
            xy_dict["x"].append(x)
            xy_dict["y"].append(y)
            xy_dict["c"].append(category)
            xy_dict["j"].append(product)
        
        xy_df = pd.DataFrame(xy_dict)
        
        ## Plot it so we can see the random coordinates that we made
        #productMap.plot_map_interactive(xy_df,data_type="random")
        productMap.plot_product_map(xy_df,data_type="random")
        
        ## Now calculate benchmarks for this df
        ss, amis, nnh = self.calculate_benchmarks(xy_df)
       
        print(f"Done calculating benchmarks for a random set of coordinates"
              f" at {utils.get_time()}! \n"
              f" Silhouette score: {ss} \n"
              f" Adjusted mutual info score:{amis} \n"
              f" nn hitrate: {nnh}")
        
        
    def plot_loss(self): 
        """Plots the results of loss files with a confidence interval"""

        print("PLotting loss graph")
        sns.set(font_scale=2,rc={'figure.figsize':(20,10)})
        #sns.set(style="whitegrid")
        ## In order to make the confidence interval, we need to reformat the
        ## data so that every individual loss value is in the list as separate
        ## number
        data = {}
        data.setdefault("batch", []) # batch
        data.setdefault("loss", []) # product
        
        for i, res in enumerate(self.loss_dict["result"]): 
            batch = self.loss_dict["batch"][i]
            
            for k in range(len(res)): # go through each list of losses
                data["batch"].append(batch)
                data["loss"].append(res[k])
            
        data = pd.DataFrame(data)
        
        ## Plot graph, aggregating y values by taking the mean
        graph = sns.lineplot(x = data["batch"], y= data["loss"],
                             estimator = np.mean)
        # Define labels
        graph.set(xlabel='Batch number', ylabel='Loss')
        plt.show()
 
    
    def get_average_cat_similarities(self, embeddings):
        """We need to calculate the average similarity between all
        category pairs, for each category"""
        
        ## Get products and categories from the dict we already made
        prod_cats = self.wi_xy[["j","c"]]
                
        ## Turn embeddings into dataframe
        embeddings = pd.DataFrame(embeddings).rename_axis("j").reset_index()
        
        ## Add categories to embeddings through left join 
        embeddings = embeddings.merge(prod_cats[["j","c"]], 
                                  how="left", left_on="j",
                                  right_on = "j") 
        # drop j, we don't want the average to be taken for this
        embeddings = embeddings.drop(columns = {"j"})

        ## Group by category and take the mean
        cat_groups = embeddings.groupby("c").mean()
        
        return cat_groups
    

    def get_sim_co_heatmaps(self):
        """To analyze similarities between products and categories, we
            first calculate the average similarity between all category pairs.
            For this, we average product vectors within each category and
            compute the dot product of the category vectors for all category
            pairs."""
        
        mean_vecs_wi = self.get_average_cat_similarities(self.wi_embeddings)
        mean_vecs_wo = self.get_average_cat_similarities(self.wo_embeddings)
        
        num_cats = len(mean_vecs_wi)
  
        ## Getting similarity scores
        similarity = np.zeros((num_cats, num_cats))
        
        for i in range(num_cats): # for each category
            for j in range(num_cats): # for each other category
                # Calculate dot product for each category pair
                similarity[i][j] = mean_vecs_wi.iloc[i] @ mean_vecs_wi.iloc[j]
        
        ## Now plot a heatmap of the similarity scores!
        visualise.correlation(similarity, colors = "plasma")
        plt.show()        

        ## Getting co-occurence scores
        occurence = np.zeros((num_cats, num_cats))
        
        for i in range(num_cats): # for each category
            for j in range(num_cats): # for each other category
                # Calculate dot product for each category pair, with the 'other' embeddings
                occurence[i][j] = mean_vecs_wi.iloc[j] @ mean_vecs_wo.iloc[i]
                occurence[j][i] = mean_vecs_wi.iloc[j] @ mean_vecs_wo.iloc[i]
        
        ## Now plot a heatmap of the co-occurence scores!
        visualise.correlation(occurence, colors = "plasma")
        plt.show()
        

    
    def load_all_files(self, base_string):
        """Should load all the files of certain type and input resuts"""
        
        print(f"loading files of type {base_string}, with {self.max_epoch}"
              f" epochs and {self.files_per_epoch} files per epoch")
        
        data = {}
        data.setdefault("epoch", []) # epoch
        data.setdefault("batch", []) # batch
        data.setdefault("result", []) # product

        batch_count = self.step_size # initialise counter
                
        for ep in range(self.max_epoch): # go through every epoch
        
            # The first batch of epochs 1+ is saved at batch number 
            # 'step_size', so we start with that
            current_batch = self.step_size 
            
            for file in range (self.files_per_epoch): # go through every file
                            
                if (ep==0 and file == 0): # Our first file is at batch one 
                

                    data["epoch"].append(0)
                    data["batch"].append(1)
                    
                    output = np.load(f'{self.embedding_dir}/{self.specific_dir}/'
                                     f'{base_string}_0_1.npy')
                    data["result"].append(output)
                      
                ## When we are not at our very first file:
                
                data["epoch"].append(ep)
                data["batch"].append(batch_count) # not current_batch?
          
                output = np.load(f'{self.embedding_dir}/{self.specific_dir}/'
                                 f'{base_string}_{ep}_{current_batch}.npy')
                                
                # for example, this could be ./output/instacart/wi_1_1000
                current_batch += self.step_size # next file is step_size higher batch
                batch_count += self.step_size
            
                data["result"].append(output)
                
                
        return data
        
    
    
    def get_favourite_products(self, cust_embeddings, n, max_customer = 3000):
        """The goal of this function is to make a dataframe with
        each customer and next to that a list of their 'favourite' 
        products"""
        
        fav_dict = {}
        
        if len(self.full_data["i"].unique()) <= max_customer:
            cust_vec = self.full_data["i"].unique()
        else: # If there are too many customers, we take a subset
            cust_vec = self.full_data["i"].unique()[0:max_customer]
            
        print(f"Getting the top {n} products for {len(cust_vec)} customers")
        
        cust_vec  = sorted(cust_vec) # Need to make sure it is sorted!!
        
        for i in tqdm(cust_vec): # for each customer, we get most frequently
        # bought products
            products = self.full_data["j"][self.full_data["i"]==i].reset_index()
            top_products = products["j"].value_counts()[:n].index.tolist()
            # Remove nan values, if people haven't bought that many products
            top_products = [prod for prod in top_products if ~np.isnan(prod)]

            fav_dict[i] = top_products
        
        return fav_dict
    
    
    def c2v_pooling(self, method = "average", seed = 1, top_products = 10,
                    max_cust = 3000):
        """Get customer embeddings by pooling product embeddings per customer.
        Options are taking the average of embeddings of all products bought,
        taking the max, taking the min, or taking the average of the top x
        products that are most frequently bought by the customer."""
   
        prod_cust = self.full_data[["j","i"]] 
        embeddings = self.wi_embeddings              
        ## Turn embeddings into dataframe
        embeddings = pd.DataFrame(embeddings).rename_axis("j").reset_index()
        
        ## Add the embeddings of each product to customer ids 
        embeddings = embeddings.merge(prod_cust[["j","i"]], 
                                  how="right", left_on="j",
                                  right_on = "j") 
        
        print(f"Getting customer vecs by {method} pooling")  
        
        ## Now pool the embeddings, based on the specified method
        if method == "average":
            ## Group by category and take the mean  
            cust_embeddings = embeddings.drop(columns = {"j"}).groupby("i").mean()
        elif method == "max":
            ## Take the max
            cust_embeddings = embeddings.drop(columns = {"j"}).groupby("i").max()
        elif method == "min":
            ## Take the min
            cust_embeddings = embeddings.drop(columns = {"j"}).groupby("i").min()
        elif method == "top_products":
            # the top n products for each customer
            fav_prod = self.get_favourite_products(embeddings, n=top_products,
                                                   max_customer=max_cust)

            # We now use dictionary comprehension to create a dictionary where
            # the key is each customer and the values are the average of 
            # the embeddings of their top n products
            tmp = embeddings.drop(columns = {"i"}).copy().set_index("j")
            cust_embeddings = {cust:np.mean(tmp.loc[fav,:]) 
                for cust,fav in fav_prod.items()}
            del tmp # make some space in working memory 
                       
            cust_embeddings = pd.DataFrame(cust_embeddings).transpose()
            
        # Take a subselection if num_cust is too large
        if len(cust_embeddings) > max_cust:
            cust_embeddings = cust_embeddings[:max_cust] # take first x rows
        
        # Get the xy coordinates
        print("Doing tsne for customer vecs")
        tsne_output = sklearn.manifold.TSNE(random_state=seed, # a seed num
                                    n_components=2, # barnes hut dimension
                                    n_iter=4000, # max iter for optimization
                                    init = 'pca',
                                    perplexity=15, # used in barnes hut
                                    angle=0.5, # used in barnes hut
                                    verbose=0).fit_transform(cust_embeddings)
        
        
        print("Putting results into dict")
            
        xy_dict = {}
        xy_dict.setdefault("x", []) #x coordinate
        xy_dict.setdefault("y", []) #y coordinate
        
        if self.data_type == "simulated_c2v":
            xy_dict.setdefault("cat_i", []) #customer category
            xy_dict.setdefault("i", []) #y coordinate


        for i in range(len(cust_embeddings)):
            x = tsne_output[i,0] 
            y = tsne_output[i,1]
                    
            ## Add the results
            xy_dict["x"].append(x)
            xy_dict["y"].append(y)
            
            if self.data_type == "simulated_c2v":
                
                cat_i = self.full_data["c_i"][self.full_data["i"]==i].unique()
                cat_i = cat_i[0]
    
                xy_dict["cat_i"].append(cat_i)
                xy_dict["i"].append(i)
                
        print(pd.DataFrame(xy_dict))
                
        if self.data_type == "simulated_c2v":
            # We can use categories to calculate benchmarks for this
            ss, amis, nnh = self.calculate_benchmarks(xy_dict,pooled =True)
          
            print(f"Done calculating benchmarks for {method} pooled customer vecs"
                  f" at {utils.get_time()}! \n"
                  f" Silhouette score: {ss} \n"
                  f" Adjusted mutual info score:{amis} \n"
                  f" nn hitrate: {nnh}")
        
        # Now make a product map
        print("plot product map for customer vecs")
        if self.data_type == "simulated_c2v":
            productMap.plot_product_map(pd.DataFrame(xy_dict),data_type = "simulated_c2v_pooled")
        else:
            productMap.plot_product_map(pd.DataFrame(xy_dict),data_type = "c2v")

            
    
if __name__ == '__main__':
    
    instacart = False
    simulated_c2v = True
    simulated = False
    
    embedding_dir = "./output"
    if instacart:    
        max_epoch = 50
        max_batch = 5000
        
        data_dir = "./instacart_2017_05_01"
        specific_dir = "instacartc50_epoch50_L30"
        data_type ="instacart"
        base_name = "instacart_cat50prod10seed1234"
    
        
        testEval = evaluationClass(base_name, data_dir, embedding_dir,
                        specific_dir,
                        data_type, max_epoch, max_batch, step_size =1000)
        
        #testEval.plot_loss() 
        #testEval.get_random_benchmarks()
        #testEval.get_sim_co_heatmaps()
        testEval.c2v_pooling(method = "top_products", top_products = 10)
    
        
    elif simulated_c2v:
        base_name = "simulated_c2v_t100_c1000"
        data_dir = "./largeData"
        specific_dir = "simulated_c2v"
        data_type = "simulated_c2v"
        
        max_epoch = 5
        max_batch = 6000
        
        testEval = evaluationClass(base_name, data_dir, embedding_dir,
                        specific_dir,
                        data_type, max_epoch, max_batch, step_size =1000)
            
        #testEval.plot_loss() 
        #testEval.get_random_benchmarks()
        testEval.c2v_pooling(method = "top_products",top_products =10)
        
    elif simulated:
        base_name = "simulated_data_t100_c1000"
        data_dir = "./largeData"
        specific_dir = "simulatedt100c1000_epoch5_L15"
        data_type = "simulated"
        
        max_epoch = 5
        max_batch = 5000
        
        testEval = evaluationClass(base_name, data_dir, embedding_dir,
                        specific_dir,
                        data_type, max_epoch, max_batch, step_size =1000)
            
        #testEval.plot_loss() 
        #testEval.get_random_benchmarks()
        testEval.get_sim_co_heatmaps()
     
            