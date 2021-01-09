# -*- coding: utf-8 -*-
"""
Implements the data preparation for the P2V-MAP framework as described 
by Gabel et al (2019). There are three steps to this: data needs to be put in
the right format for the skipgram model and infrequent products need to be 
removed, positive training samples need to be created by combining all basket
products into center-context pairs and lastly negative samples need to be 
created.

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import numpy as np                  
import pandas as pd
from itertools import permutations
from collections import defaultdict
from tqdm import tqdm

# Own modules:
import visualise
import utils


class data_preprocessor:
    """This class contains methods to preprocess and 
    visualise raw basket data for the Skip-Gram model"""
     
    def __init__(self, input_directory, output_directory,
                 main_or_train,  split=True, test_data=None,validation_data=None, 
                 basket_id_name = "basket_hash", product_id_name = "j", 
                 visualisation = True, n_min=20):
        """Initialisation for the data processor class"""
        
        #-------------------------INITIALISATION-----------------------
        self.input_directory = input_directory # location of input files
        self.output_directory = output_directory # location of split output files
        self.main_or_train = main_or_train # either this is a train dataset,
        # or this is one dataset that still needs to be split!!     
        self.split = split #boolean for whether a datafile needs to be split
        self.test_data = test_data # testing dataset
        self.validation_data = validation_data #validation dataset 
        self.basket_id_name = basket_id_name 
        self.product_id_name = product_id_name
        self.visualisation = visualisation # boolean whether or not to show graphs
        self.n_min = n_min # minimum frequency of products
        
        
        #-------------------------DATA PREP-----------------------

        ## We split the data if necessary
        if split:
            self.split_data(self.main_or_train, indir= self.input_directory,
                            outdir= self.output_directory,
                            id_variable = self.basket_id_name)     
            
        elif (test_data==None or validation_data==None): 
            message = ("when split is set to False, you should input a validation, "
                       "train and test set separately")
            raise myCustomizedError(message)
            
        else: # read the three files into dataframes
            self.train_df = pd.read_csv(f"{self.input_directory}/{self.main_or_train}.csv")
            self.test_df = pd.read_csv(f"{self.input_directory}/{self.test_data}.csv")
            self.validation_df = pd.read_csv(f"{self.input_directory}/{self.validation_data}.csv")       
            
        
        ## Next step is to remove the least frequently occuring products, to make
        ## the training better.
        if visualisation:
            visualise.product_counts(self.train_df,self.n_min) # visualise before and after
            self.train_df, self.to_remove = self.remove_infrequent_products(
                self.train_df, "trainset",self.n_min)
            visualise.product_counts(self.train_df,self.n_min)
        else:
            self.train_df, self.to_remove = self.remove_infrequent_products(
                self.train_df, "trainset",self.n_min)
        
        ## Now we remove the same products from the test and validation sets as what 
        ## we removed from the training set
        self.test_df, self.to_remove = self.remove_infrequent_products(dataset_df=self.test_df, name="testset",to_remove= self.to_remove)
        self.validation_df, self.to_remove = self.remove_infrequent_products(dataset_df=self.validation_df, name="validationset",to_remove= self.to_remove)
 
        ## Reset indices after removal 
        if len(self.to_remove)>0:
            print("resetting indices after having removed some products")
            self.train_df = self.reset_indexing(self.train_df)
            self.test_df = self.reset_indexing(self.test_df)
            self.validation_df = self.reset_indexing(self.validation_df)
            
        ## Save the datasets
        if split: # If we actually had to split the datasets, otherwise not necessary
            utils.save_df_to_csv(self.train_df, self.output_directory, 
                                 f"{self.main_or_train}_split_train", 
                                 add_time=False)
            utils.save_df_to_csv(self.test_df, self.output_directory,
                                 f"{self.main_or_train}_split_test",
                                 add_time=False)
            utils.save_df_to_csv(self.validation_df,self.output_directory, 
                                 f"{self.main_or_train}_split_validation", 
                                 add_time=False)
            
            print(f"Split files saved to {self.output_directory}!")
        

    def reset_indexing(self,df,cat_name = "c"):    
        """We need to reset the indexes to start at 0 and end at the
        number of products, because otherwise the training and the mapping won't
        work"""
        temp_df = df.sort_values(by=cat_name)
        unique_prods = temp_df[self.product_id_name].unique()
        
        df = df.replace({self.product_id_name: {unique_prods[i] : i
                                 for i in range(len(unique_prods))}})
        return df
    
                
    def split_data(self, dataset, indir="./largeData", outdir ="./largeData", 
                   id_variable = 'basket_hash', split1=0.6, split2= 0.2):
        """Splits a single dataset into train, test and validation set.
        The default produces a 60%, 20%, 20% split for training, 
        validation and test sets"""
        
        print(f"Beginning with splitting datasets, at {utils.get_time()}.")
        
        ## Read into csv file
        self.full_df = pd.read_csv(f"{indir}/{dataset}.csv")
        
        ## We want to split it by basket id as we don't want products from
        ## the same basket to end up in different files,
        
        ids = self.full_df[id_variable].unique()
        np.random.seed(1234) # For consistent results through different runs
        np.random.shuffle(ids) # Returns nothing, just shuffles the list and overwrites `ids`

        ## Split the ids into train, test and validation sets
        ids_train = ids[:int(split1*len(ids))]
        ids_test = ids[(int(split1*len(ids))+1):(int(split1*len(ids))+int(split2*len(ids))+1)]
        ids_validation = ids[(int(split1*len(ids))+int(split2*len(ids))+1):]


        ## Now we select subsamples of the full dataframe into individual dataframes
        self.train_df = self.full_df.loc[self.full_df[id_variable].isin(ids_train)]
        self.test_df = self.full_df.loc[self.full_df[id_variable].isin(ids_test)]
        self.validation_df = self.full_df.loc[self.full_df[id_variable].isin(ids_validation)]
        

        ## Save the dataframes separately as csv files, so we can check them out & reuse
        utils.save_df_to_csv(self.train_df, outdir, 
                              f"{dataset}_split_train", add_time=False)
        utils.save_df_to_csv(self.test_df, outdir, f"{dataset}_split_test",
                             add_time=False)
        utils.save_df_to_csv(self.validation_df, outdir, 
                             f"{dataset}_split_validation", add_time=False)
    
        print(f"done with splitting at {utils.get_time()}")
    


    def remove_infrequent_products(self, dataset_df, name, n_min=20, product_ids ='j', 
                                   basket_ids = 'basket_hash', to_remove=None):
        """The goal of this function is to remove products from a dataset that
        occur in fewer than n_min baskets. Two options: when no to_remove df is
        given, we make our own, otherwise we remove products that are already defined"""  
                
        if to_remove is None: # If no series is given, we make our own
        
            product_counts = get_product_counts(dataset_df, product_ids, 
                                                basket_ids) 
            to_remove = product_counts[product_ids][product_counts[basket_ids] < n_min]
            
            prod_removed = len(to_remove)
            print(f'Removing products from {name}: {prod_removed} products occur in fewer than {n_min} baskets')
        
        else:
            prod_removed = len(to_remove)
            print(f'Removing {prod_removed} products from {name}')
        
        rows_before = len(dataset_df)
        smaller_dataset_df = dataset_df.loc[~dataset_df[product_ids].isin(to_remove)]
        rows_after = len(smaller_dataset_df)
        row_removed = rows_before-rows_after
                
        print(f'Done! {row_removed} rows removed from {name}')
        return smaller_dataset_df, to_remove



    def get_prepped_data_processor(self, dataset_df, dataset_name, outdirec,
                                   outname,
                                   basket_id_name = "basket_hash", 
                                   product_id_name = "j", n_neg = 20, power=0.75,
                                   batch_size = 10000, domain=2**31-1):
        """Make a p2vstream class for a dataset"""

        if dataset_df is not None:
            return pair_generator(dataset_df,dataset_name, outdirec, outname
                                  )




class pair_generator(object): 
    """ This class contains methods to make center-context pairs"""

    def __init__(self, dataset, dataset_name,outdir,outname, basket_id_name = "basket_hash", 
                 product_id_name = "j", n_neg = 20, power=0.75, domain=2**31-1
                 ):
        """ Initialised a model to process data which can be used for the 
        product2vec model. Makes center-context pairs, and also has a method
        for doing negative sampling"""
        
        #-------------------------INITIALISATION-----------------------
        
        print(f"getting pair generator for {dataset_name}")
        
        self.dataset = dataset # The dataset containing baskets that
        # we want to make center-context pairs for
        self.dataset_name = dataset_name
        self.outdir = outdir
        self.outname = outname # name that we wish to save it as
        self.basket_id_name = basket_id_name # variable of the basket id column
        self.product_id_name = product_id_name # variable of the product id column
        self.n_neg = n_neg # the number of negative samples to take
        self.power = power # the power of negative samples distribution
                   
        #----------------------PREPARING TO MAKE PAIRS-------------------------
        
        ## The dictionary is for iterating through later
        print("making product dictionary")
        self.product_dict = defaultdict(list)
        for index, row in tqdm(self.dataset.iterrows()):
            self.product_dict[row[basket_id_name]].append(row[product_id_name])
        
        
        ## Get dfs of product counts and product 
        self.product_counts = get_product_counts(dataset,product_id_name,
                                              basket_id_name)
        self.products = self.product_counts[product_id_name] 
        
        ## Things needed for negative sampling
        self.domain = domain #integer range, used in sampling
        
        ## Cumulative count table to map products to unique integers:
        self.count_table = self.get_cumulative_counts()

 
 
    def get_center_contex_pairs(self):
        """The goal of this method is to create all of center, positive and
        negative contexts for every product, and save them to files"""
        
        # Initialise empty lists 
        center_products = []
        pos_context_sample = []
        neg_context_sample = []
        
        print(f"Generating center-context pairs for {self.dataset_name}," 
              f"at {utils.get_time()}. n_neg is {self.n_neg}")
        
        for i, (basket, products) in tqdm(enumerate(self.product_dict.items())):

            ## get permutations of length 2 for the products in this basket
            permutation_list = permutations(products, 2)
            center_context_pairs = list(map(list, list(permutation_list)))

            ## Now get the negative samples
            if self.n_neg > 0:
                neg_samples = self.get_negative_samples(np.array(center_context_pairs))
    
                ## Add the products to the specific vectors
                for n, cc_pair in enumerate(center_context_pairs):
                    center_products.append(cc_pair[0])
                    pos_context_sample.append(cc_pair[1])
                    neg_context_sample.append(neg_samples[n])
                    
            else: # add products 
                for cc_pair in enumerate(center_context_pairs):
                    center_products.append(cc_pair[0])
                    pos_context_sample.append(cc_pair[1])


        ## Now we save the vectors into csv files        
        print(f'Saving to file, {len(center_products)} tuples created')                
        
        # We can put add_time as 'True' so that files created earlier don't 
        # get overwritten
        utils.save_df_to_csv(pd.DataFrame(center_products), self.outdir, 
                             f"{self.outname}_center_products_{self.dataset_name}",
                             add_time=False)
        utils.save_df_to_csv(pd.DataFrame(pos_context_sample), self.outdir, 
                             f"{self.outname}_pos_context_{self.dataset_name}",
                             add_time=False)
        utils.save_df_to_csv(pd.DataFrame(neg_context_sample), self.outdir, 
                             f"{self.outname}_neg_context_{self.dataset_name}",
                             add_time=False)   
        

        print("Done! Saved center,context and negative products to datafiles"
              f" at {utils.get_time()}")
        
        return (np.array(center_products), np.array(pos_context_sample), 
                np.array(neg_context_sample))



    def get_negative_samples(self, pos_context):
        """Method that takes some random products which are not
        in the same basket as our center product as negative samples, 
        for each center product"""
        
        ## Creates array and sets all values to -1:
        neg_samples = np.zeros((len(pos_context), self.n_neg))-1
        comparison_matrix = np.zeros((len(pos_context), self.n_neg))-1

        ## If any value in the matrix is still -1, we keep going
        while (np.any(neg_samples == comparison_matrix)):
            
            ## how many samples do we need?
            new_sample_index = (neg_samples == -1)    
            n_draws = np.sum(new_sample_index)
            
            ## Get random integers between 0 and our range
            draws = np.random.randint(0, self.domain, n_draws)
            # We use this integer to select products from our cumulative
            # count table, which maps product ids to unique integers
            negative_sample_index = np.searchsorted(self.count_table, draws)
            neg_sample_products = self.products[negative_sample_index]
            
            ## Now put the negative sample products into the matrix
            neg_samples[new_sample_index] = neg_sample_products
            
            ## We want to avoid collisions, so we reset those products to -1       
            for neg_sample in neg_samples:
                 if neg_sample in pos_context: # avoid collisions
                     i = list(neg_samples).index(neg_sample)
                     neg_samples[i] = -1 # set it back to -1, so we can keep searching

        return neg_samples.astype(int)


    def get_cumulative_counts(self):
        """Method to make a cumulative count table, from 0 until self.domain.
        This way we can map each product to a unique integer"""
        cum = np.array(self.product_counts[self.basket_id_name], dtype=float) ** self.power
        counts = np.cumsum(cum / sum(cum))
        count_table = ((counts * self.domain).round())
        return count_table
    


class Error(Exception):
    """Base class for exceptions."""
    pass

class myCustomizedError(Error):
    """Exception raised for errors.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message=None,expression=None):
        self.expression = expression
        self.message = message


def get_product_counts(dataset_df, product_id_name='j',
                              basket_id_name="basket_hash"):
    """This counts in how many unique baskets a product appears.
    we do not count if a product is bought a lot at once, only
    in how many different baskets it appears"""

    product_counts_unique = dataset_df.groupby(product_id_name)[basket_id_name].nunique()
            
    return product_counts_unique.reset_index()
        



if __name__ == "__main__":
    
    # Change these variables to make the code do stuff
    simulation_data_prep = False
    instacart_data_prep = False
    check_results = False
    
    
    if simulation_data_prep:
        indir = "./largeData"
        outdir= "./largeData"
        filename = "simulated_data_t100_c1000"
        
      
        sim_process = data_preprocessor(indir,outdir,filename, visualisation =False)
     
        # Pak de pair generator, voor nu alleen even voor train file
        train_pairmaker = sim_process.get_prepped_data_processor(sim_process.train_df, 
                                                             dataset_name = "train",
                                                             outdirec = outdir,
                                                             outname = filename)
        # Output will be saved to files
        center_prod, pos_prod, neg_prod = train_pairmaker.get_center_contex_pairs() 
       
        
    ## This is just for some testing, checking how long the resulting files are  
    if check_results:
        print("Checking the resulting files")
        
        readFromFile = False
        if readFromFile == False and simulation_data_prep == False:
            print("Run the dataprep first before you can check resulting"
                  "vectors, or select output files to check")
        
        if readFromFile: 
            indir = "./largeData"
            
            center_prod = pd.read_csv(f"{indir}/simulated_data_center_products_train.csv")
            pos_prod = pd.read_csv(f"{indir}/simulated_data_pos_context_train.csv")
            neg_prod = pd.read_csv(f"{indir}/simulated_data_neg_context_train.csv")
            
        print(len(center_prod))
        print(len(pos_prod))
        print(len(neg_prod))
        
        
 