# -*- coding: utf-8 -*-
"""
This file contains methods to visualise datasets (outside of the methods to
make the product mappings)

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Own code imports:
import utils
import dataPrep



def correlation(correlation_matrix, cent = 0, colors = "coolwarm"): 
    """Visualises a matrix in a heatmap, with the numbers labeled"""
    
    sns.set(rc={'figure.figsize':(23.4,16.54)})
    graph = sns.heatmap(correlation_matrix, center=cent, annot = True, 
                        cmap=colors) 
    plt.show()



def datafile(df,week_name ="t",product_id_name="j",
             basket_id_name = "basket_hash", price_name = "price",
             cat_name = "c"):
    """This function is to create various visualisations and statistics
    of a product basket datafile. Not useful for instacart data, as it 
    does not have price """
    
    print(f"visualising data, at {utils.get_time()}")
        
    #------------------------------------------------------------------
    print("Number of products per basket - summary statistics:")
    df_group = df.groupby(basket_id_name).size()
    print(df_group.describe())
    
    #------------------------------------------------------------------
    ## Make a co-occurence matrix to see how often categories occur together
    ## with products
    # print("plotting co-occurence")
    # co_mat = pd.crosstab(df[cat_name], df[product_id_name])
    # sns.set(rc={'figure.figsize':(23.4,16.54)})
    # graph1=sns.heatmap(co_mat)
    # plt.show()
    #------------------------------------------------------------------
    
    ## Visualisation of which categories are bought most often??
    print("plotting category counts")
    graph2=sns.countplot(x=cat_name, data = df)
    plt.show()
    
    #------------------------------------------------------------------
    ## Visualisation of which products are bought most often??
    # print("plotting total product counts")
    # graph3=sns.countplot(x=product_id_name, data = df)
    # plt.show()
    #------------------------------------------------------------------
    
    ## Average price
    print("plotting price per category")
    graph4=sns.barplot(x=cat_name,y=price_name,data=df)
    plt.show()
    
      
    #------------------------------------------------------------------
    
    ## Make a visualisation of product frequencies, mapped against price
    print("plotting frequencies vs price")
    sns.set(font_scale=2,rc={'figure.figsize':(20,15)})
    
    df_pricegrouped = df.groupby(price_name)[[product_id_name,cat_name]]
    sizes = df_pricegrouped.size()
    
    hue = True # hue = True assumes unique prices per product!
       
    if hue: 
        newsizes = pd.DataFrame.from_dict({"frequency":sizes,
                                           "prices": sizes.index})  
        newsizes.reset_index() 
        
        ## This function does a left join, like in SQL, so we can get categories 
        ## as well
        newsizes = newsizes.merge(df[[price_name,cat_name]].drop_duplicates(),
                                              how="left", left_on="prices",
                                              right_on = price_name,)  
       
        #The data needs to be categorical for 'hue' to work!
        newsizes[cat_name] = newsizes[cat_name].astype('category')
        
        #newsizes = newsizes.sort_values(cat_name) # sort by cat name
        
        graph5 = sns.scatterplot(x ="prices", y= "frequency", 
                                 data=newsizes, hue=cat_name, s = 80) #s is marker size

         # do I need order = cat_name??
    else:
        graph5 = sns.scatterplot(data=sizes)
    
    graph5.set(xlabel='Price', ylabel='Frequency in baskets')
    
    plt.show()
    
    #------------------------------------------------------------------

    
def product_counts(dataset_df,
                   n_min=20,product_id_name="j",
                   basket_id = "basket_hash", cat_name = "c",
                   cust_id = "i"):
    """Visualises product counts in a dataframe"""

    print(f"visualising product counts, at {utils.get_time()}.")        
    product_counts = dataPrep.get_product_counts(dataset_df, product_id_name)      
    
    ## This function does a left join, like in SQL, so we can get categories as well
    product_counts = product_counts.merge(dataset_df[[product_id_name,cat_name]].drop_duplicates(),
                                          how="left", on=product_id_name)
    
    #The data needs to be categorical for 'hue' to work!
    product_counts[cat_name] = product_counts[cat_name].astype('category')
        
    sns.set(font_scale=2,rc={'figure.figsize':(60,25)})
    graph = sns.barplot(x=product_id_name, y=basket_id,
                        hue= cat_name, dodge=False,
                        data=product_counts, edgecolor=None)
    
    if n_min is not None: # put a horizontal line for the minimum product counts
        graph.axhline(n_min) 
    
    # Define labels and rotate to make them readable
    graph.set(xlabel='Product id', ylabel='Frequency')
    graph.set_xticklabels(graph.get_xticklabels(),rotation=90,
                          horizontalalignment='right', fontweight='light',
                          fontsize=10)
    graph.legend(loc='upper right', ncol=1) 
    
    annotate = False     
    if annotate:
        ## We want to label the bars with the height, to see what the exact counts are
        for p in graph.patches:
            value = p.get_height()
            # locations of where to put the labels
            x = p.get_x() + p.get_width() / 2 - 0.05 
            y = p.get_y() + p.get_height()
            graph.annotate(value, (x, y), size = 10)
    
    plt.show()
    
    print("Done visualising counts!")

  
def instacart_df(dataset_df, n_min=20, product_id_name="j",
                 basket_id = "basket_hash", cat_name = "c",
                 cust_id = "i"):
    
    print("Visualising instacart dataframe")
    
    print("Plotting counts")
    # dit was prod_id
    counts = dataset_df[[product_id_name,"prod_name","product_frequency","aisle"]]
    counts = counts.sort_values(by="aisle") # Order by category
    
    sns.set(font_scale=3,rc={'figure.figsize':(60,25)})
    sns.set_style("whitegrid")
    fig, graph = plt.subplots()
    sns.barplot(x=product_id_name, y="product_frequency",
                        hue= "aisle", dodge=False,
                        data=counts, edgecolor=None, ax=graph)
    if n_min is not None: # put a horizontal line for the minimum product counts
        graph.axhline(n_min) 
    # Define labels and rotate to make them readable
    graph.set(xlabel='Product id', ylabel='Frequency')
    # graph.set_xticklabels(graph.get_xticklabels(),rotation=90,
    #                       horizontalalignment='right', fontweight='light',
    #                       fontsize=10)
    graph.set(xticklabels=[])
    graph.legend(loc='upper left',bbox_to_anchor=(1, 1), ncol=2)
    #  bbox_to_anchor=(1.25, 0.5), <- if we want it next to the graph,
    #  change location to 'upper left' and use bbox
    
 
    
    plt.show()
    
  

if __name__ == "__main__":
   
    visualise_simulation_data = True
    visualise_product_counts = False
    visualise_instacart = False
    visualise_gabel = False
    

    if visualise_simulation_data:
        ## Visualising some aspects of the output datafile 
        ## (make sure to use right filename)
        filename = "simulated_data_t100_c1000.csv"
        direc = "./largeData"
        df_test_visualisation = pd.read_csv(f"{direc}/{filename}")
        
        datafile(df_test_visualisation) 
   
    if visualise_product_counts:  
        ## Use the full simulation dataset just to test if the function works
        filename = "simulated_data_t100_c1000.csv"
        direc = "./largeData"
        df_test_visualisation = pd.read_csv(f"{direc}/{filename}")
        
        product_counts(df_test_visualisation) 
    
    if visualise_instacart:  # Does NOT work right now
        ## Use the full simulation dataset just to test if the function works
        filename = "instacart_cat50prod10seed1234.csv"
        direc = "./instacart_2017_05_01"
        df_test_visualisation = pd.read_csv(f"{direc}/{filename}")
        
        instacart_df(df_test_visualisation) 
        
    if visualise_gabel:
        filename = "gabel_baskets_train_final.csv"
        direc = "./largeData"
        df_test_visualisation = pd.read_csv(f"{direc}/{filename}")
        
        product_counts(df_test_visualisation) 
        
