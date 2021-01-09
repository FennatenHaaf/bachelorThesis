# -*- coding: utf-8 -*-
"""
Provides methods to process raw instacart data into a single datafile
containing a subset of categories and products from the original file

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""
                
import pandas as pd

# Own modules:
import visualise
import utils
import dataPrep


class instacart_processor:

    def __init__(self,indir,outdir,output_name, 
                 cat_subselect = 50, prod_subselect = 10,
                 custExtension = False,
                 n_cust = 300,
                 reset_indexes = True,
                 seed = 1234, n_min=20,
                 cust_id = "i", 
                 prod_id_name ="j",
                 basket_id_name = "basket_hash", 
                 cat_name = "c"): 
        """This method should process the instacart_2017_05_01 raw dataset,
        into a form to use with the other dataPrep methods"""
        
        print(f"Processing instacart dataset, at {utils.get_time()}")
        
        #-------------------------INITIALISATION-----------------------
        self.indir = indir
        self.outdir = outdir
        self.output_name = output_name
        self.cat_subselect = cat_subselect
        self.prod_subselect = prod_subselect
        self.seed = seed # for the random sample of categories
        self.n_min= n_min
        self.cust_id = cust_id
        self.prod_id_name = prod_id_name
        self.basket_id_name = basket_id_name
        self.cat_name = cat_name
        self.reset_indexes = reset_indexes
        
        #---------------------READING IN RAW DATA---------------------
       
        #We treat the aisles as 'categories', but alternatively we could also
        # just consider the departments. There are 21 departments and 134 aisles
        self.aisles_df = pd.read_csv(f'{indir}/aisles.csv')
        self.departments_df = pd.read_csv(f'{indir}/departments.csv')
        # Order_products links products to basket ids.
        # There is a train set and a prior set that we can use, prior is larger 
        # than train as it contains more orders per customer. 
        self.orders_df = pd.read_csv(f'{indir}/order_products__prior.csv') 
        # The orders file links orders to customer ids:
        self.cust_df = pd.read_csv(f'{indir}/orders.csv') 
        # The products file links products to names and aisles:
        self.product_info_df = pd.read_csv(f'{indir}/products.csv') 
        
    
        #---------------------COMBINING INTO ONE FILE--------------------- 
        ## Create a dataframe with basket hash and product id    
        self.final_df = pd.DataFrame.from_dict({"basket_hash":self.orders_df["order_id"],
                                            "j": self.orders_df["product_id"]})  
    
        ## Add customer IDs by doing a left join    
        self.final_df = self.final_df.merge(self.cust_df[["order_id","user_id"]], 
                                  how="left", left_on="basket_hash",
                                  right_on = "order_id",) 
        
        ## Add product names, aisle ids and department ids by left join 
        self.final_df = self.final_df.merge(self.product_info_df[["product_id",
                                                             "product_name",
                                                             "aisle_id",
                                                             "department_id"]], 
                                            how="left", left_on="j",
                                            right_on = "product_id",) 
        ## Add aisle names by left join
        self.final_df = self.final_df.merge(self.aisles_df[["aisle_id","aisle"]], 
                                  how="left", left_on="aisle_id",
                                  right_on = "aisle_id",) 
        
        ## Add deparment names by left join
        self.final_df= self.final_df.merge(self.departments_df[["department_id",
                                                                "department"]], 
                                  how="left", left_on="department_id",
                                  right_on = "department_id",) 
    
        ## Rename and drop some things, so it is in the right format
        self.final_df = self.final_df.rename(columns={"aisle": cat_name,  # could also rename aisle
                                 "user_id": cust_id,
                                  "j":"prod_id",
                                  "product_name":"j"})
        self.final_df = self.final_df.drop(["product_id","order_id"],1)
        
        
        #---------------------TAKING SUBSELECTIONS--------------------- 
        num_unique = len(self.final_df["j"].unique())
        print(f"There are {num_unique} products in the full instacart dataset,"
              f" we will now be taking subselections")
        
        if cat_subselect is not None:
            self.final_df = self.category_subselect(self.final_df, 
                                                    self.cat_subselect,
                                                    self.seed)
            
        if prod_subselect is not None:
            self.final_df = self.product_subselect(self.final_df)  
            
        
        #---------------------SAVING RESULTS--------------------- 
        
        # First rename before saving so that c and j are the integer values 
        self.final_df = self.final_df.rename(columns={cat_name: "aisle",  # could also rename aisle
                                                      "aisle_id": cat_name,
                                                      "j":"prod_name",
                                                      "prod_id":self.prod_id_name})
        
        ## RESET INDICES SO THAT IT WORKS WITH THE MAPPING WHICH NEEDS THE INDICES
        ## TO MATCH WITH THE ROWS
        if self.reset_indexes:
            print("resetting indexing")
            self.reset_indexing()
        
        print("saving to csv")
        utils.save_df_to_csv(self.final_df, self.outdir, 
                              self.output_name, add_time = False )      
        print(f"Finished and output saved, at {utils.get_time()}")
        
    
    
    def reset_indexing(self):    
        """We need to reset the indexes to start at 0 and end at the
        number of products, because otherwise the training and the mapping won't
        work"""
        temp_df = self.final_df.sort_values(by=self.cat_name)
        unique_prods = temp_df[self.prod_id_name].unique()
        
        self.final_df = self.final_df.replace(
            {self.prod_id_name: {unique_prods[i] : i
                                 for i in range(len(unique_prods))}})


    
    def category_subselect(self, final_df, cat_subselect = 30, seed = 1234):
        """Method to take a random sample of n unique categories and returns
        only the purchases that belong to those categories"""
        
        print(f"Randomly selecting {cat_subselect} categories (aisles) from"
              " the instacart dataset, but not those that appear in too"
              " few baskets")
        ## Take a subselection of prodcuts from only a few categories
        ## (without subselection, the final file is 2 GB!)
        
        # Before taking a sample, we should remove the categories that appear
        # in very few baskets out of consideration 
        print("counting cats")
        cat_counts = final_df.groupby(self.cat_name)[self.basket_id_name].nunique()
        cat_counts.reset_index()
        print(cat_counts)
        min_count = cat_counts.min()
        print(min_count)
        
        # We would like on average at least n_min of each product per category 
        min_baskets = (self.prod_subselect*self.n_min) 
        print(f"making to_remove, min_baskets = {min_baskets}")

        to_remove = cat_counts[cat_counts < min_baskets]  

        smaller_aisles_df = self.aisles_df.loc[~self.aisles_df["aisle"].isin(to_remove)]
        ## In practice, there aren't actually any categories that get removed...
        
        ## We also don't want the category "missing" in our dataframe, because
        ## They are vague
        smaller_aisles_df = smaller_aisles_df.loc[self.aisles_df["aisle"] != "missing"]
        smaller_aisles_df = smaller_aisles_df.loc[self.aisles_df["aisle"] != "more household"]
       
        print(f"Taken {len(to_remove)} categories out of consideration," 
              f" as they appeared in fewer than {min_baskets} baskets")
        
        # Now we can randomly select a sample (random_state is a seed,
        # for consistent results)
        cat_subset = smaller_aisles_df["aisle"].sample(n=cat_subselect,
                                               random_state = seed)
        final_df = final_df.loc[final_df[self.cat_name].isin(cat_subset)]
        
        print("Done, the following categories were selected:")
        print(final_df[self.cat_name].unique())
        
        return final_df



    def product_subselect(self, final_df, prod_subselect = 10):
        """Takes a subsample of an instacart dataframe by only selecting
        the products that are in the top n most frequent products for their
        respective category"""  
        # NOTE: not sure if this works when there are fewer than n products
        # in a category
    
        print(f"Taking a subset of {prod_subselect} most frequent products"
              " per category")
        product_counts = dataPrep.get_product_counts(final_df)
        product_counts = product_counts.rename(columns={self.basket_id_name: "product_frequency"})
       
        ## Do a left join to add the product frequencies to the dataframe
        # print(product_counts.head())
        final_df = final_df.merge(product_counts, 
                                  how="left", left_on=self.prod_id_name,
                                  right_on = self.prod_id_name) 
        
        products_to_keep = pd.DataFrame()
        
        ## Now loop over categories and take out the n most common products
        # print(final_df[prod_id_name].head())
        
        for i,category in enumerate(final_df[self.cat_name].unique()):
            
            print(category)
            products = final_df.loc[final_df[self.cat_name].isin([category])]
            products = products[self.prod_id_name]
            products = products.value_counts()
            products = products.head(prod_subselect) # take only the top n products
            products.reset_index()
    
            products_to_keep = products_to_keep.append(products)
            
        
        final_df = final_df.loc[final_df[self.prod_id_name].isin(products_to_keep)]
      
        # print("Done! Now visualising the resulting dataframe")
        # visualise.instacart_df(final_df, self.n_min)
        
        return final_df
    
    
    
if __name__ == "__main__":

    instacart_data_prep= True
    
    if instacart_data_prep: 
        
        indir = "./instacart_2017_05_01"
        outdir= "./instacart_2017_05_01"
        filename = "instacart_cat50prod10seed1234" # remember to adjust filename
        
        instacart_data = instacart_processor(indir,outdir,filename,
                                             cat_subselect=50,prod_subselect=10,
                                             seed = 1234)
        instacart_df = instacart_data.final_df
        

    
    
    