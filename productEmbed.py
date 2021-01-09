# -*- coding: utf-8 -*-
"""
Runs a modified skip gram model as described by Gabel et. al (2019)

@author: Fenna ten Haaf
Written for the Econometrics & Operations Research Bachelor Thesis
Erasmus School of Economics
"""
import os                           
import numpy as np                  
import pandas as pd                 
from tqdm import tqdm
import tensorflow as tf             

# My own modules
import utils
import productMap


class P2V_inputs_processor:
    """Processes the vectors of center, positive and negative products that 
    will be put into batches and used to train the skip gram model """
    

    def __init__(self, dataset_name, center_vec, pos_vec, neg_vec, from_file=False,
                 indir=None, n_neg=20, step_size=1000):
        """Initialise processSkipGramInputs graph """
        
        print(f"processing {dataset_name} vectors for SkipGram, at {utils.get_time()} ")
        
        #-------------------------INITIALISATION-----------------------
        # self.outdir = outdir # where to save the batch files??
        self.dataset_name = dataset_name
        self.center_vec = center_vec # Vector (or file!) of center products
        self.pos_vec = pos_vec # Vector (or file) of poitive context products
        self.neg_vec = neg_vec # Array (or file) of negative sample products
        self.from_file = from_file # Whether the inputs need to be read from
        # files still
        self.step_size = step_size 
        self.n_neg = n_neg
        
        
        #-----------------------------GET VECTORS-----------------------
         
        if from_file:
            self.center_prod = pd.read_csv(f"{indir}/{center_vec}.csv").to_numpy()
            self.pos_prod = pd.read_csv(f"{indir}/{pos_vec}.csv").to_numpy()
            self.neg_prod = pd.read_csv(f"{indir}/{neg_vec}.csv").to_numpy()
        else:
            self.center_prod = self.center_vec
            self.pos_prod = self.pos_vec
            self.neg_prod = self.neg_vec
        
        
        #-------------------MAKE SOME THINGS FOR TRAINING----------------
        
        center_df = pd.DataFrame(self.center_prod)
        self.num_products = len(center_df[0].unique())
        
        self.batch_num = 0
        self.starting_index = 0
    
        # length of the context vector per product
        self.context_num = self.n_neg+1 
              
        # This is how many batches there should technically end up being
        # - should be rounded down
        self.num_batches = (len(self.center_prod) // self.step_size) 
        
        
        
        
    def batch_generator(self): 
       """generator for making batches from the inputs. """ 
    
       for b in range(self.num_batches+1):

            ## intialise batch vectors
            self.batch_center = np.zeros(self.step_size)
            self.batch_positive = np.zeros(self.step_size)
            self.batch_negative = np.zeros(self.step_size)
            
            ## If we reach the end of the center products vector
            if self.starting_index + self.step_size > len(self.center_prod):
               
                self.batch_center = self.center_prod[self.starting_index:len(self.center_prod)]
                self.batch_positive = self.pos_prod[self.starting_index:len(self.center_prod)]
                self.batch_negative = self.neg_prod[self.starting_index:len(self.center_prod)]
                
                print(f"Done making batches, goal was {self.num_batches} and result"
                      f" {self.batch_num} of length {self.step_size}")
                
            ## Otherwise we can just keep going like normal
            else:
        
                self.batch_center = self.center_prod[self.starting_index:(self.starting_index+self.step_size)]
                self.batch_positive = self.pos_prod[self.starting_index:(self.starting_index+self.step_size)]
                self.batch_negative = self.neg_prod[self.starting_index:(self.starting_index+self.step_size)]
                
                ## Update the starting index
                self.starting_index += self.step_size
             
            # Now we return the batch - cool thing about a generator is that
            # we can call it and it will go onto the next iteration every time
            # instead of doing the whole for loop in one go
            yield [self.batch_center.astype(np.int32), 
                   self.batch_positive.astype(np.int32),
                   self.batch_negative.astype(np.int32)] 
            
            # This is just for testing
            #self.save_batch_csv()
            self.batch_num+=1
        
        
    def save_batch_csv(self):
        """Save batch file, just for testing if it works"""
        #print("done! Going to save data into files")
        # Could add function that puts batch_num into the name??
        
        outdir = "./largeData"
        df_center = pd.DataFrame(self.batch_center)
        df_context = pd.DataFrame(self.batch_context)
        df_labels = pd.DataFrame(self.batch_labels)
        
        utils.save_df_to_csv(df_center, outdir, "batchfile_center")
        utils.save_df_to_csv(df_context, outdir, "batchfile_context")
        utils.save_df_to_csv(df_labels, outdir, "batchfile_labels")
        print(f"Done! Files saved at {utils.get_time()}")
    
        
    def reset_indexes(self):
        """Resets the numbers used in the batch generator"""
        self.starting_index = 0
        self.batch_num = 0
        
        
        
        
class P2Vimplementation: 
    """ The implementation of Product2Vec (gabel(2019)) using tensorflow"""
   
    def __init__(self, L, data_streamer, data_type,
                 batch_size, outpath,
                 save_interval = 1000):
        """Initialiser for P2Vimplementation class"""
        
        #-------------------------INITIALISATION-----------------------
        
        self.L = L # dimension size of the output embeddings
        self.data_type = data_type # Which dataset (gabel,instacart or simulation)
        self.batch_size = batch_size # number of training samples per batch
        self.save_interval = save_interval #save training results every save_interval-th step
    
        #-------------------DEFINING SOME OTHER VARIABLES--------------
        
        self.data_streamer = data_streamer 
        self.num_prod = data_streamer.num_products # how many products to make vectors for
        self.n_neg = data_streamer.n_neg
        
        # Initialise counters
        self.epoch_count = 0
        self.batch_num = 0
        
        ## Making the output directory
        self.outpath = os.path.join(outpath, f"{self.data_type}")
        self.result_file_pattern = os.path.join(self.outpath, '%s_%d_%d.npy')
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
      
        #-----------------------START THE MODEL-----------------------
        
        print(f"Initialising model for {self.data_type} data at {utils.get_time()},"
              f" with {self.num_prod} products. Output is embeddings of dimension"
              f" {self.L}")
        
        self.initialise_model()
            
            
    def initialise_model(self, seed = 501):
        """Function to define the model inputs and layers"""
    
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed) 
        
        tf.compat.v1.reset_default_graph() # Clears the default graph stack 
        tf_config_proto = tf.compat.v1.ConfigProto()
        tf_config_proto.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=tf_config_proto)
            

        #-----------------BUILD THE TF GRAPH----------------
        
        ## define the inputs
        tf.compat.v1.disable_eager_execution() 
        
        self.center_input = tf.compat.v1.placeholder(dtype=tf.int32, name='center', shape=[None])
        self.pos_context_input = tf.compat.v1.placeholder(dtype=tf.int32, name='context', shape=[None])
        self.neg_context_input = tf.compat.v1.placeholder(dtype=tf.int32, 
                                                          name='neg_samples', 
                                                          shape=[None, self.n_neg]) 
        self.tf_learn_rate = tf.compat.v1.placeholder(dtype=tf.float32, 
                                                          name='learning_rate', 
                                                          shape=[])
    
        # Initialise with a random constant
        initializer_wi = tf.compat.v1.truncated_normal_initializer(stddev=0.08,
                                                                   seed = self.seed)
        initializer_wo = tf.compat.v1.truncated_normal_initializer(stddev=0.08,
                                                                   seed = self.seed)

        # Defining the shape of the center embedding matrix (the input vectors)
        self.wi = tf.compat.v1.get_variable(
                name='wi',
                shape=[self.num_prod, self.L], # should be 150 by 15, for gabeldata
                dtype=tf.float32,
                initializer=initializer_wi
                )

        # output weight matrix
        self.wo = tf.compat.v1.get_variable(
                name='wo',
                shape=[self.num_prod, self.L], 
                dtype=tf.float32,
                initializer=initializer_wo
                )
     
        ## Defining center, positive and negative samples layers
        self.wi_center = tf.nn.embedding_lookup(self.wi, self.center_input)
        self.wo_positive_samples = tf.nn.embedding_lookup(self.wo, self.pos_context_input)
        self.wo_neg_samples = tf.nn.embedding_lookup(self.wo, self.neg_context_input)

        self.logits_pos = tf.einsum('ij,ij->i', self.wi_center, self.wo_positive_samples)
        self.logits_neg = tf.einsum('ik,ijk->ij', self.wi_center, self.wo_neg_samples)

        ## we now add a global bias value, beta_0 in the paper
        self.global_bias = tf.compat.v1.get_variable(
                name='bo',
                dtype=tf.float32,
                initializer=tf.constant(-3.0, dtype=tf.float32)
                )

        # Add the bias term to the logits
        self.logits_pos = self.logits_pos + self.global_bias
        self.logits_neg = self.logits_neg + self.global_bias
   
        ## Defining the loss for positive and negatice
        # loss for positive context
        loss_pos = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.logits_pos),
                logits=self.logits_pos)

        # loss for negative context
        loss_neg = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.logits_neg),
                logits=self.logits_neg )

        # total (average) batch loss
        loss = tf.reduce_mean( tf.divide( # sum all losses
                    loss_pos + tf.reduce_sum(loss_neg, axis=1),
                    (self.n_neg + 1)))

        # save loss
        self.loss = loss
        
       
        #-------------------INITIALISE OPTIMIZER--------------------
       
        ## We use the Adam optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(self.tf_learn_rate, 
                                                     beta1 =0.9, 
                                                     beta2 = 0.999,
                                                     epsilon=1e-08)
        
        self.step_global = tf.Variable(0, name="global_step")
        self.optimize_train = optimizer.minimize(self.loss, global_step=self.step_global)

     
        #------------------INITIALISE VARIABLES--------------------
       
        tf.compat.v1.global_variables_initializer().run(session=self.session)



    def train(self, num_epoch, learning_rate):
        """ A function to perform the training steps. For every epoch, we pass 
        through the whole dataset again to train even more. """

        print(f"Starting to train network, at {utils.get_time()}")
        
        self.learning_rate_epoch = learning_rate

        # Loop over all the epochs
        for epoch in range(num_epoch):

            # Reset and activate the batch generator
            self.batch_num = 0
            self.data_streamer.reset_indexes()
            data_stream = self.data_streamer.batch_generator()
            
            # loss cache
            self.loss_cache = []
            
            print(f"Training model, for epoch {epoch}."
                  f" {self.data_streamer.num_batches} iterations expected.")

            # iterate through all training examples in training data set
            
            for x, train_data in tqdm(enumerate(data_stream)): 
                
                center = train_data[0].reshape(-1)
                context = train_data[1].reshape(-1)
                neg_samples = train_data[2]
                       
                self.batch_num += 1 # update counter

                ## Now we do one training step
                               
                loss, _ = self.session.run([self.loss, self.optimize_train],
                                           feed_dict={self.center_input: center,
                                                      self.pos_context_input: context,
                                                      self.neg_context_input: neg_samples,
                                                      self.tf_learn_rate: learning_rate})
                self.loss_cache.append(loss)
                
                # Save training loss for the very first batch
                if (self.batch_num==1) and self.epoch_count == 0:
                    self.save_current_output() # didn't save learning rate before
 
                # Save output for training data, every time we reach the 
                # save_interval number
                if self.batch_num % self.save_interval == 0:
                    mean_loss =  np.mean(self.loss_cache)
                    print(f" mean loss {mean_loss} at batch {self.batch_num}")
                    self.save_current_output() #TODO: this didn;t save learning rate before
                    
            # Save output at the end of the epoch
            self.save_current_output() # This did save learning rate before

            # update epoch counter
            self.epoch_count += 1
            
            

    def save_current_output(self):
        """ Saves the embedding outputs, loss and learning rate """

        file_base = os.path.join(
                self.outpath,
                '%s_%d_%d.npy' % ('%s', self.epoch_count, self.batch_num)
                )

        np.save(file=file_base % 'train_loss', arr=np.array(self.loss_cache))
        self.loss_cache = [] # reset the loss cache after saving it

        wi = self.session.run(self.wi) # input embedding (v in the gabel paper)
        np.save(file=file_base % 'wi', arr=wi)
        
        wo = self.session.run(self.wo) # output embedding (w in the gabel paper)
        np.save(file=file_base % 'wo', arr=wo)

    
    
if __name__ == '__main__':
    
    # Testing it with the gabel dataset
    
    indirec = "./largeData"
    outdirec = "./output"
    centername = "original_baskets_center_products_train"
    posname= "original_baskets_pos_context_train"
    negname= "original_baskets_neg_context_train"
    num_epochs = 5
    data_source = "gabel"
        
    # batch_steps = 1000
    # train_stream = P2V_inputs_processor(dataset_name= "train",
    #                                      center_vec=centername,
    #                                      pos_vec=posname, 
    #                                      neg_vec=negname,
    #                                      from_file=True,
    #                                      indir = indirec,
    #                                      n_neg=20, 
    #                                      step_size=batch_steps)
      
    # p2v = P2Vimplementation( L= 15,
    #                          data_type = data_source,
    #                          batch_size = batch_steps,
    #                          outpath = outdirec,
    #                          save_interval = 1000,
    #                          data_streamer = train_stream)

    # p2v.train(num_epoch=5, learning_rate = 0.0005) 

    # # Now visualise it
    
    # print("Doing tsne for gabel data")
    # #data_indir = "./largeData"
    # #output_indir = "./results/p2v-map-example"
    # output_indir = "./output/gabel"
    
    # batches = train_stream.num_batches
    # epochs = num_epochs - 1
    
    # instacart_map = productMap.product_mapping(output_indir, data_type = "gabel",
    #                                            epoch = epochs, batch = batches,
    #                                            seed = 1)
