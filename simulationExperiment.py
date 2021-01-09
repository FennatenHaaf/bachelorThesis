# -*- coding: utf-8 -*-
"""
This file contains methods for a simulation of shopping baskets as described 
by Gabel et al (2019). The output could be used with the P2V-Map framework
in order to test its effectiveness. 

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# My own files
import utils
import visualise


class shopper_simulation:
    """This class contains methods to simulate customers with shopping baskets"""
     
    def __init__(self, t, num_categories, num_products, num_consumers,
                 product_pref, c2v_sim = False, subsample=True):
        """ First we initialise the class with the parameters in the model: 
            in week t, consumer i chooses product j from category c. Each category 
            consists of J(c) products, and a product belongs to exactly one category 
        """
        
        #-------------------------INITIALISATION-----------------------
        
        print(f"Initialising simulation, at {utils.get_time()}."
              f" Parameters: num_weeks = {t}, num_cat = {num_categories},"
              f" num_prod = {num_products}, num_cons = {num_consumers}")
        
        self.subsample = subsample # Boolean for if we take a subsample
        self.t = t # The number of weeks
        self.num_categories = num_categories # The number of categories
        self.num_products = num_products # The number of products per category
        self.num_consumers = num_consumers # The number of consumers
        self.product_pref = product_pref # The standard deviation
        # of the product preference heterogeneity, necessary to compute the
        # category-specific covariance matrix S(c).
        self.c2v_sim = c2v_sim # This is if we want to create consumers with specific
        # preferences
        
        #----------------------CREATING CATEGORIES---------------------   
        print("Creating the categories")
        
        ## The regular price of products in each category is sampled from a 
        ## log-normal distribution with a mean of .5 and a standard deviation 
        ## of .3.
        self.category_prices = np.zeros(self.num_categories)
        self.category_prices = self.create_category_prices(0.5, 0.3)
        
        self.products_dict = {}
        for cat in tqdm(range(self.category_prices.size)):
            self.product_prices = self.create_product_prices(cat)
            self.products_dict[cat] = self.product_prices
            
        print("creating sigmas")
        ##We build MNP covariance matrices Σ(c) for each category by
        ##calculating Σ(c) = (τ(c)I(c))Ω(c)(τ(c)I(c)),
        ##where the random correlation matrices Ω(c) are built using the 
        ##Vine method (Lewandowski,Kurowicka, and Joe 2009).
        self.omegas = self.create_omega() #random correlation matrix
        self.sigmas = self.create_sigma() #MNP covariance matrix
        
    
        #--------------------------CONSUMERS--------------------------
        print("creating consumers")
        self.consumers = []
        
        ## This is to create consumers based on 10 different categories
        if self.c2v_sim:
            matrix = pd.read_excel('data/custParameters.xlsx', header=None)
            gammas = matrix.iloc[0:20,:] # first 20 rows
            
            # just to check it out - colours could be more greyscale friendly?
            visualise.correlation(gammas, cent=-0.5, colors = "Greys") 
            p_sensitivities = matrix.iloc[20] # last row in the dataset

            cust_id = 0
            for c_i in range(10):
                # we need to select the i'th row
                this_gammas = gammas.iloc[:,c_i]
                this_p_s = p_sensitivities.iloc[c_i]
                
                for i in tqdm(range(self.num_consumers//10)):
                    cons = consumer(prod_prices_dict = self.products_dict,
                                   sigma = self.sigmas, C=self.num_categories,
                                   gamma = this_gammas,
                                   J_c=self.num_products, 
                                   p_sensitivity=this_p_s, 
                                   dev=1,
                                   id=cust_id,
                                   c_id=c_i)
                    cust_id+=1 # update cust id
                    self.consumers.append(cons)
            
             
        ## This is to create consumers where gamma and product sensitivity
        ## is the same for everyone
        else:
            for i in tqdm(range(self.num_consumers)): #tqdm helps track progress
               cons = consumer(prod_prices_dict = self.products_dict,
                               sigma = self.sigmas, C=self.num_categories,
                               gamma=-0.5,
                               J_c=self.num_products, p_sensitivity=2, dev=1,id=i)
               self.consumers.append(cons)
          
        ## Now we simulate the baskets for each customer and put it in a dictionary
        self.simulated_data = self.simulate_shopping()
        
         
 
    def create_category_prices(self, mean, std):
        """ the regular price of products in each category, p(c), is sampled 
        from a log-normal distribution with a mean of .5 and a standard 
        deviation of .3"""
        prices = np.zeros(self.num_categories)
        for i in range(prices.size):
            prices[i] = np.random.lognormal(mean, std)
        return prices
    

    def create_product_prices(self,category):
        """Product prices are sampled from a uniform distribution between
        p/2 and 2p with p being the category prices"""
        product_prices = np.zeros(self.num_products)
        category_price = self.category_prices[category]
        for i in range(product_prices.size):
            product_prices[i] = np.random.uniform(category_price / 2, category_price * 2)
        return product_prices
    
    
    def create_omega(self):
        """Method to sample the positive partial correlations in Ω(c) from a 
        Beta(.2, 1) distribution, with the vine method"""
        omegas = {}
        for i in range(self.num_categories):
            omegas[i] = self.vine_method(self.num_products, 0.2, 1)
        return omegas


    def create_sigma(self):
        """Method to calculate the MNP covariance matrix, with
        Σ(c) = (τ(c)I(c))Ω(c)(τ(c)I(c))"""
        sigmas = {} # dit is een dictionary
        I = np.eye(self.num_products) # J(c)-dimensional identity matrix
        for i, omega in self.omegas.items():
            sigma_c = (self.product_pref * I) @ omega @ (self.product_pref * I)
            sigmas[i] = sigma_c
        return sigmas
    
    
    def vine_method(self, dimensions, beta1, beta2):
        """Implements the Vine method (Lewandowski, Kurowicka, and Joe 2009)"""
    
        partial_correlations = np.zeros((dimensions, dimensions))
        omega_c = np.eye(dimensions)

        for k in range(dimensions):
            for i in range(k + 1, dimensions):
                partial_correlations[k, i] = np.random.beta(beta1, beta2)
                p = partial_correlations[k, i]
                for l in range(k, 0, -1):
                    p = p * np.math.sqrt(
                        (1 - partial_correlations[l, i] ** 2) * (1 - partial_correlations[l, k] ** 2)) + \
                        partial_correlations[l, i] * partial_correlations[l, k]
                omega_c[k, i] = p
                omega_c[i, k] = p
        return omega_c


    def simulate_shopping (self):
        """This method is to simulate the baskets for each consumer """
        dict_data = defaultdict(list)
        basket_id = 0
        
        print(f'Creating baskets, at {utils.get_time()}')
        for week in tqdm(range(self.t)):

            ## Now we create a basket for every customer for every week
            for consumer in self.consumers:
                simulated = consumer.make_basket(basket_id, week)
                for key,value in simulated.items():
                    dict_data[key].extend(value) # add it to the dictionary
                basket_id += 1
        return dict_data
    
        
    
    def output_to_csv(self, outdir = './largeData', visualisation=False):
        """This method is to return the data from the simulated customers in
        csv files"""
        
        # Structure of the datafile is as follows: first it shows the 
        # n customers in week 0, then the n customers in week 2, etc, t
        # times. The file shows i,cat,j,price,t,basket_id.
        
        if self.subsample:  # in this case we don't have to split the dataset
            data = self.simulated_data
            df = pd.DataFrame.from_dict(data)
            if self.c2v_sim:
                filename_string = f"simulated_c2v_t{self.t}_c{self.num_consumers}"
            else:
                filename_string = f"simulated_data_t{self.t}_c{self.num_consumers}"
            
            #Save file with time added so it does not overwrite anything
            utils.save_df_to_csv(df,outdir,filename_string, add_time =False)
        
        #else:
            # Using t=200 and n=10 000 results in more than 18 million purchases,
            # I can save it but it takes up a lot of space
            
        if visualisation:
            visualise.datafile(df)
        
        return df
    




class consumer(object):
    """The consumer class, which models a shoppper that can buy one product from
    a category each week, and buys 2 of that product half of the time"""

    def __init__(self, sigma, prod_prices_dict, C, gamma, 
                 J_c, p_sensitivity, dev, id, c_id = None):
        """ Initialisation of the Consumer class. 
        """
        #-------------------------INITIALISATION-----------------------
        self.dev = dev # Standard deviation for error term in MNP
        self.p_sensitivity = p_sensitivity # price sensitivity
        self.J_c = J_c # Number of products per category
        self.gamma = gamma # base utility for category purchase incidence,
        # this is either one value or it is a list!
        self.C = C # Number of categories
        self.id = id
        self.c_id = c_id
        self.prod_prices_dict =  prod_prices_dict # Dictionary with product prices
        # per each category
        self.sigma = sigma #category-specific covariance matrix
        
    
        # We need to get the manual correlation matrix which is used in the 
        # category choice
        self.correlation_matrix = self.get_MVP_correlation()    

    
    def make_basket(self, basket_id, week):
        """ This function is to create a basket of products for a specific
        customer in a specific week """
            
        basket_dict = utils.get_standard_basket_dict()
        if self.c_id is not None:
            basket_dict.setdefault("c_i", []) #customer category
        
        # First, the categories are chosen that products will be bought from
        categories = self.choose_categories()
        
        for i, choice in enumerate(categories):
            if choice == 1: # they chose the category
                # The consumer now chooses which product to buy
                products, price = self.buy_products(i)
                for x, product in enumerate(products):
                    basket_dict["i"].append(self.id) # customer id
                    basket_dict["basket_hash"].append(basket_id) # basket id
                    basket_dict["c"].append(i) # category
                    basket_dict["j"].append(product) # product id
                    basket_dict["price"].append(price[x]) # price
                    basket_dict["t"].append(week) # week
                    if self.c_id is not None:
                        basket_dict["c_i"].append(self.c_id) #customer category
                    
        return basket_dict
    
    
    
    def choose_categories(self):
        """Given a latent variable z_ict =
        g_c + e_ict that depends on the base utility of category purchase
        g_c and a (correlated) error term e_ict * MVN(0, O), category
        purchase incidence is given by y_ict = 1 if z_ict>0 and y_ict = 0 
        otherwise:"""
        categories = np.zeros(self.C)
        
        # the correlation matrix is the one that we made manually
        error = np.random.multivariate_normal(np.zeros(self.C), 
                                              self.correlation_matrix)
       
        for i in range(categories.size):   
            # Gamma is a base utility
            if self.c_id is not None: # in this case gamma is a list
                 utility = self.gamma.iloc[i] + error[i] 
            else: # in this case gamma is -0.5 for everything
                utility = self.gamma + error[i]  
            if utility > 0:
                categories[i] = 1  
        return categories


    def buy_products(self, category):
        """ Product utility is the sum of a base utility a_ij , 
        the (dis)utility incurred by paying price Þ_j
        (multiplied by price sensitivity b(c)), and a random error term"""
        
        max_util = (float('-inf'), -1)
        second_max_util = (float('-inf'), -1)
        
        # creating the base utility for this category: we assume that
        # a_ij ~ MVN(0,Sigma(c)) with Sigma(c) a category-specific covariance
        # matrix
        
        base_utility = np.random.multivariate_normal(np.zeros(self.J_c),
                                                     self.sigma[category])
        
        # Get the product prices
        product_prices = self.prod_prices_dict[category]

        randint = np.random.uniform(0,1)
        #50 percent chance that 2 products are bought instead of only one.
        if (randint <= 0.50):
            # Now the utility is calculated with a_ij - b*p_j + random error
            for i in range(self.J_c):
                
                error = np.random.normal(0, self.dev)
                utility = base_utility[i] - self.p_sensitivity * product_prices[i] + error
                               
                if utility > max_util[0]:
                    second_max_util = max_util
                    max_util = utility, i
                elif utility > second_max_util[0]:
                    second_max_util = utility, i
            return [max_util[1] + category * self.J_c, second_max_util[1] + category * self.J_c], [product_prices[max_util[1]], product_prices[second_max_util[1]]]

        else:
            for i in range(self.J_c):
                
                utility = base_utility[i] - self.p_sensitivity * product_prices[i] + np.random.normal(0, self.dev)
                if utility > max_util[0]:
                    max_util = utility, i
            return [max_util[1] + category * self.J_c], [product_prices[max_util[1]]]
   
    
    
    def get_MVP_correlation(self):
        """ Function to get the MVP correlation matrix which was made 
        manually in Excel. It captures (random) purchase incidence
        correlation across categories
        """    
        matrix = pd.read_excel('data/MVP_correlation_matrix.xlsx', header=None)
        MVP_correlation = matrix.to_numpy()
        return MVP_correlation
    
    
    





if __name__ == "__main__":
   
    # Change these variables to make the code do stuff
    run_c2v_sim = True
    run_simulation = False
    run_visualisation = False
    visualise_correlations = False
     
    if run_c2v_sim:
        simulation = shopper_simulation(num_products=15, 
                              num_categories=20, t=100, num_consumers=1000,
                              product_pref=2, c2v_sim = True,
                              subsample=True
                              )
        
        print("Done! Now converting output to csv files")
        df_output = simulation.output_to_csv(visualisation=True) 
     
    if run_simulation:
        ## Gabel et al(2019) samples baskets for I=10000 consumers and T=200 weeks.
        ## However, this results in more than 18 million entries, which is impractical.
        ## We take a subsample of 100 weeks and 1000 consumers
        simulation = shopper_simulation(num_products=15, 
                              num_categories=20, t=100, num_consumers=1000,
                              product_pref=2, subsample=True)
        
        print("Done! Now converting output to csv files")
        if run_visualisation:
            df_output = simulation.output_to_csv(visualisation=True) 
        else:
            df_output = simulation.output_to_csv()
            
            
    if visualise_correlations:    
        ## visualising manual MVP correlation
        matrix = pd.read_excel('data/MVP_correlation_matrix.xlsx', header=None)
        MVP_correlation = matrix.to_numpy()
        visualise.correlation(MVP_correlation) 
    
        ## MNP correlation matrix for first category, just to check
        visualise.correlation(simulation.omegas[0]) 
    
    
