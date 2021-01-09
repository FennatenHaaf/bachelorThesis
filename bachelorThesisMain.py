# -*- coding: utf-8 -*-
"""
This file runs the methods necessary to produce results for the p2v-map
bachelor thesis. You can run the files individually for individual results,
but you can also just run this file to run everything in one go. 

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""

import pandas as pd

# My own modules:
import utils
import simulationExperiment
import createInstacartData as CID
import dataPrep
import productEmbed
import productMap
import benchmarkAndPool


class runBachelorThesis(object):
    """This class runs the code for the bachelor thesis """

    def __init__(self, data_type, base_name, raw_data_dir, processed_data_dir,
                 embedding_dir):
        """Initialise the runBachelorThesis code"""
        
        #-------------------------INITIALISATION-----------------------
        
        self.data_type = data_type # the data source, can be instacart, gabel, simulated or simulated_c2v
        self.base_name = base_name # forms the base of the filenames for this data type
        self.raw_data_dir = raw_data_dir # Location of raw data inputs
        self.processed_data_dir= processed_data_dir # location of processed data inputs
        self.embedding_dir = embedding_dir # where the embeddings are saved
        
        assert (self.data_type in {"instacart", "simulated", "gabel", "simulated_c2v"}), \
            "valid data types are: instacart, simulated, simulated_c2v or gabel"
            
            
        #----------------------DEFINE NAMES OF DATASETS-------------------
       
        if self.data_type == "gabel": # We already have this one split up
            self.train_name = "gabel_baskets_train_final"
            self.test_name = "gabel_baskets_test"
            self.validation_name = "gabel_baskets_validation"
        else:
            self.train_name = f"{self.base_name}_split_train"
            self.test_name = f"{self.base_name}_split_test"
            self.validation_name = f"{self.base_name}_split_validation"
           
        #---------------------DEFINING NAMES OF PRODUCTS-----------------
      
        self.centername = f"{self.base_name}_center_products_train"
        self.posname= f"{self.base_name}_pos_context_train"
        self.negname= f"{self.base_name}_neg_context_train"
    
    
    def run_data_prep(self):
        """ Runs the data preparation necessary for the skip gram model.
        
        Three steps are undertaken: the input file is split into three sets,
        infrequent products are removed and lastly every product is paired with
        positve context and negative context products, which are saved into 
        csv files"""    
        
        if self.data_type == "gabel": # We already have this one split up
             process = dataPrep.data_preprocessor(self.raw_data_dir,
                                                  self.processed_data_dir,
                                                  main_or_train = self.train_name, 
                                                  split= False,
                                                  test_data = self.test_name,
                                                  validation_data = self.validation_name,
                                                  visualisation = False)
        else: # We need to split the data
            process = dataPrep.data_preprocessor(self.raw_data_dir,
                                                 self.processed_data_dir,
                                                 main_or_train = self.base_name,
                                                 visualisation = False)
        
        # Now create the center_context pairs for the train set, currently
        # validation and test are not implemented
        train_pairmaker = process.get_prepped_data_processor(process.train_df, 
                                                     dataset_name = "train",
                                                     outdirec = self.processed_data_dir,
                                                     outname = self.base_name)
        center_prod, pos_prod, neg_prod = train_pairmaker.get_center_contex_pairs()
                
        
        
    def train_embedding_and_visualise(self, epoch_num, L_size,
                                      max_batch_num, benchmark = True,
                                      batch_steps=1000):
         """Train embeddings based on center-context pairs and visualise
         the results. If benchmark is set to true, we also calculate
         benchmark results and compare it to scores for random results.
         We additionally plot the loss graph and show similarity scores"""
            
         # Process the inputs for embeddings
         train_stream = productEmbed.P2V_inputs_processor(dataset_name= "train",
                                                           center_vec=self.centername,
                                                           pos_vec=self.posname, 
                                                           neg_vec=self.negname,
                                                           from_file= True,
                                                           indir = self.processed_data_dir,
                                                           n_neg=20, 
                                                           step_size=batch_steps)
         # Now train the embeddings
         p2v = productEmbed.P2Vimplementation(L = L_size,# embedding dimension
                                               data_type = self.data_type, # gabel, instacart or simulated
                                               batch_size = batch_steps, # how many pairs in one batch
                                               outpath = self.embedding_dir, # where the embeddings will be saved
                                               save_interval = 1000, # how often to save
                                               data_streamer = train_stream)
         
         p2v.train(num_epoch = epoch_num, learning_rate = 0.0005) 
         
         #We can immediately visualise the results
         productMap.product_mapping(base_name = self.base_name, # the base string
                                    data_dir = self.processed_data_dir, # where the data is located
                                    embedding_dir = self.embedding_dir, # where embeddings are saved
                                    data_type = self.data_type,
                                    epoch = epoch_num, 
                                    batch = train_stream.num_batches+1,
                                    seed = 1)
         if benchmark:
             ## Benchmark the output files
             testEval = benchmarkAndPool.evaluationClass(base_name = self.base_name,
                                        data_dir = self.processed_data_dir,
                                        embedding_dir = self.embedding_dir, 
                                        specific_dir = self.data_type, # this is standard output place
                                        data_type= self.data_type, 
                                        max_epoch = epoch_num,
                                        max_batch = max_batch_num, 
                                        step_size =1000)
             
             #testEval.get_random_benchmarks() # Get a random benchmark for comparison
             testEval.plot_loss() # Plot loss graph of everything
             #testEval.c2v_pooling(method = "average")
            
             if self.data_type == "simulated":
                 testEval.get_sim_co_heatmaps()
  
         
         
    def only_visualise(self, direc, epoch_num, batch_num, 
                       interactive_map=True, benchmark = True,):
        """Method to visualise embeddings located in a specific output 
        directory. If we set interactive_map to False, it is output in the 
        console with a legend. Otherwise, the output is a html object with 
        labels visible if you hover over the markers. If benchmark is set to
        true, we also calculate benchmark results and compare it to scores 
        for random results. We additionally plot the loss graph and show 
        similarity scores"""
        
        product_map = productMap.product_mapping(base_name = self.base_name, # the base string
                                                 data_dir = self.processed_data_dir, # where the data is located
                                                 embedding_dir = self.embedding_dir, # where embeddings are saved
                                                 data_type = self.data_type,
                                                 epoch = epoch_num, 
                                                 batch = batch_num,
                                                 otherdir = direc, # non-standard directory (possibly)
                                                 seed = 1,
                                                 interactive = interactive_map)
        if benchmark:
             ## Benchmark the output files
             testEval = benchmarkAndPool.evaluationClass(base_name = self.base_name,
                                        data_dir = self.processed_data_dir,
                                        embedding_dir = self.embedding_dir, 
                                        specific_dir = direc, # specific location
                                        data_type= self.data_type, 
                                        max_epoch = epoch_num,
                                        max_batch = batch_num, 
                                        step_size =1000)
             
             #testEval.get_random_benchmarks() # Get a random benchmark for comparison
             testEval.plot_loss() # Plot loss graph of everything
             testEval.c2v_pooling(method = "average")
             
             if self.data_type == "simulated":
                 testEval.get_sim_co_heatmaps()


def run_simulation(num_weeks,num_cons,run_visualisation=True, run_c2v_sim = False):
    """A method to run the simulation. Make sure to change the parameters 
    in this function before running, to get different results.
    
    The result of this function is a datafile/csv file, containing 
    products, basket ids, categories, prices and week numbers. 
    """
    ## Gabel et al(2019) samples baskets for I=10000 consumers and T=200 weeks.
    ## However, this results in more than 18 million entries, which is impractical.
    ## We take a subsample of 100 weeks and 1000 consumers
    simulation = simulationExperiment.shopper_simulation(num_products=15, 
                          num_categories=20, t=num_weeks, 
                          num_consumers=num_cons,
                          product_pref=2, c2v_sim=run_c2v_sim, subsample=True)
    
    print("Done! Now converting output to csv files")
    df_output = simulation.output_to_csv(visualisation=run_visualisation)

    return df_output



def print_end_message(start_time):
    """Prints message to indicate end of the run, plus it shows time passed
    since the beginning"""
    
    end_time = utils.get_time()
    time_lapsed = utils.get_time_diff(start_time,end_time)
    print(f"Bachelor thesis code ended at {end_time}."
          f" Total time lapsed: {time_lapsed}")
    
    
    
    
    

if __name__ == "__main__":
   
    # ------------------------- SELECT DATA TO USE -------------------------
    
    use_instacart = True
    c2v = False
    use_simulation = False
    use_gabeldata = False
    
 
    if not use_simulation and not use_instacart and not use_gabeldata and not c2v:
        print("first set the right variables to true to make the code do stuff."
              " Make sure to check what you are actually running")
    else:
        start = utils.get_time() # Save the starting time of this run 
        print(f"Welcome! Running the bachelor thesis code at {start}")
     
    #-------------------C2V SIMULATED (PER CAT) + RESULTS--------------------
    # Extension 1: pooling vector data, by training on the
    # whole dataset and using that to get embeddings for individual customers  
    if c2v: 
        print("Running code for the simulated dataset with different"
              " customer categories")
    
        ## Which parts of the code do we want to run?
        run_sim = False # Not necessary to run, we already have the results
        run_dataPrep = False
        run_embeddings_from_file = False
        only_visualise_mapping = True
        
        ## First create the data 
        if run_sim:
            print("starting simulation")
            df_sim = run_simulation(num_weeks = 100, num_cons =1000,
                                    run_c2v_sim = True)
        
        ## Now we can run the code
        filename = "simulated_c2v_t100_c1000" # remember to adjust filename to simulation
        c2vcode = runBachelorThesis(data_type = "simulated_c2v",
                                      base_name =  filename,
                                      raw_data_dir = "./largeData",
                                      processed_data_dir = "./largeData",
                                      embedding_dir = "./output")
         
        if run_dataPrep: # Time: around 16 minutes 
            print("Running the data preparation")
            c2vcode.run_data_prep() 
            # 3 products are filtered out
            # more than 6 585 000 tuples are created
            
        epochs = 5
        maxbatch = 6000
    
        if run_embeddings_from_file: # Time: only around 1 minutes per epoch 
            c2vcode.train_embedding_and_visualise(epoch_num=epochs,
                                                    L_size=15, 
                                                    max_batch_num = maxbatch,
                                                    benchmark = True)  
            
        if only_visualise_mapping: # only visualise a specific file, don't run anything
            directory = "simulated_c2v" # change this to whatever non-standard directory
            
            c2vcode.only_visualise(direc = directory, 
                                   epoch_num=epochs, 
                                   batch_num = maxbatch,
                                   benchmark = True,
                                   interactive_map = False)
            
        print_end_message(start)
            
  
        
     #-------------------------INSTACART + RESULTS-------------------------
    
    if use_instacart:
        
        print("Running code for the instacart dataset")
        
        create_data = False # Not necessary to run, we already created the file
        data_prep = False
        run_embeddings_from_file = False
        only_visualise_mapping = True
        
        indir = "./instacart_2017_05_01"
        outdir = "./instacart_2017_05_01"
        filename = "instacart_cat50prod10seed1234"
        # Remember to adjust instacart base name based on how many
        # categories and what seed etc. you want to use
            
        ## First we need to create the data
        if create_data: # Time: around 3 - 6 minutes      
            instacart_data = CID.instacart_processor(indir,outdir,filename,
                                                     cat_subselect=50,
                                                     prod_subselect=10,
                                                     reset_indexes = True,
                                                     seed = 1234)
            instacart_df = instacart_data.final_df
    
        
        ## Now we can run the code
        instacode = runBachelorThesis(data_type = "instacart",
                                      base_name =  filename,
                                      raw_data_dir = indir,
                                      processed_data_dir = outdir,
                                      embedding_dir = "./output")
        
        if data_prep: # Time: around 16-40 minutes, depending on cat num
        
            print("Running the data preparation")
            instacode.run_data_prep()
         
            # Note 1: 0 products filtered out

            # Note 2: because of subsetting, the order baskets are fairly small
            # for this data. The product map won't give the full picture
            
        epochs = 50
        maxbatch = 5000
    
        if run_embeddings_from_file: # Time: only around 1 minutes per epoch 
            instacode.train_embedding_and_visualise(epoch_num=epochs,
                                                    L_size=30, 
                                                    max_batch_num = maxbatch,
                                                    benchmark = True)  
            
        if only_visualise_mapping: # only visualise a specific file, don't run anything
            directory = "instacartc50_epoch50_L30" # change this to whatever non-standard directory
            
            instacode.only_visualise(direc = directory, 
                                     epoch_num=epochs, 
                                     batch_num = maxbatch,
                                     benchmark = True,
                                     interactive_map = True)
                    
        print_end_message(start)
        
        
    ## --------------------- SIMULATION + RESULTS -------------------------
    
    if use_simulation:
        print("Running code for the simulated dataset")
    
        ## Which parts of the code do we want to run?
        run_sim = False # Not necessary to run, we already have the results
        run_dataPrep = False # Not necessary to run
        run_embeddings_from_file = False
        only_visualise_mapping = True
        
        ## First create the data 
        if run_sim:
            print("starting simulation")
            df_sim = run_simulation(num_weeks = 100, num_cons =1000)
        
        ## Now we can run the code
        filename = "simulated_data_t100_c1000" # remember to adjust filename to simulation
        simulationcode = runBachelorThesis(data_type = "simulated",
                                      base_name =  filename,
                                      raw_data_dir = "./largeData",
                                      processed_data_dir = "./largeData",
                                      embedding_dir = "./output")
         
        if run_dataPrep: # Time: around 16 minutes
            print("Running the data preparation")
            simulationcode.run_data_prep()
            # 8 products are filtered out
            # more than 5 230 000 tuples are created
            
            
        epochs = 5
        maxbatch = 5000
    
        if run_embeddings_from_file: # Time: only around 1 minutes per epoch 
            simulationcode.train_embedding_and_visualise(epoch_num=epochs,
                                                    L_size=15, 
                                                    max_batch_num = maxbatch,
                                                    benchmark = True)  
            
        if only_visualise_mapping: # only visualise a specific file, don't run anything
            directory = "simulatedt100c1000_epoch5_L15" # change this to whatever non-standard directory
            
            simulationcode.only_visualise(direc = directory, 
                                     epoch_num=epochs, 
                                     batch_num = maxbatch,
                                     benchmark = True,
                                     interactive_map = False)
            
        print_end_message(start)
        
        
        
    #------------------------GABEL(2019) DATA---------------------------
        
    # Using the data that the github of gabel et al(2019) provided, 
    # just to check how we are doing on a dataset that we can be certain
    # is suitable for running P2V-MAP on
    if use_gabeldata:
        print("Running code for the gabel(2019) dataset")
        
        create_data = False # Not necessary to run, we already created the file
        data_prep = False
        run_embeddings_from_file = True
        only_visualise_mapping = False
        
        ## Create the data first
        if create_data: # Time: 10 seconds
            train = "gabel_baskets_train"
            master = "gabel_master"
            final_name = "gabel_baskets_train_final"
        
            print("making a new datafile that also has categories")
            # We want to add 'c' to the main file 
            train_df = pd.read_csv(f'{indir}/{train}.csv')
            master_df = pd.read_csv(f'{indir}/{master}.csv')
            
            ## Add cat IDs by doing a left join    
            train_df = train_df.merge(master_df[["j","c"]], 
                                  how="left", left_on="j",
                                  right_on = "j",) 
            ## Save resulting datafile
            utils.save_df_to_csv(train_df, outdir, 
                                 final_name, add_time = False )
        
        
        ## Now we can run the rest of the code
        gabelcode = runBachelorThesis(data_type = "gabel",
                                      base_name = "gabel_baskets",
                                      raw_data_dir = "./largeData",
                                      processed_data_dir = "./largeData",
                                      embedding_dir = "./output")
        
        if data_prep: # Time: 14 minutes (14:06)
            print("Running the data preparation")
            gabelcode.run_data_prep()
            # Note: 0 products are filtered out, around 4 770 000 tuples 
            # are created
                         
            
        epochs = 5
        maxbatch = 4000
    
        if run_embeddings_from_file: # Time: only around 1 minutes per epoch 
            gabelcode.train_embedding_and_visualise(epoch_num=epochs,
                                                    L_size=15, 
                                                    max_batch_num = maxbatch,
                                                    benchmark = True)  
            
        if only_visualise_mapping: # only visualise a specific file, don't run anything
            directory = "gabel" # change this to whatever non-standard directory
            
            gabelcode.only_visualise(direc = directory, 
                                     epoch_num=epochs, 
                                     batch_num = maxbatch,
                                     benchmark = True,
                                     interactive_map = True)        
    
        print_end_message(start)
        

        
        
        
        
        
            