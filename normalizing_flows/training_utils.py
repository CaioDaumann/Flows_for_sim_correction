# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd
import zuko

# importing other scripts
import plot.plot_utils        as plot_utils

#early stopping class
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Class that will handle most of the heavy work to perform the corrections!
# This class should be able to train and evaluate the flow on Z->ee samples
# But also perform the needed valdiaitons on Z->mumugamma and Diphoton samples 
class Simulation_correction():
   
    def __init__(self):

        # Checking if cuda is avaliable
        print('Checking cuda avaliability: ', torch.cuda.is_available())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        #reading important tensors
        self.read_saved_tensor()
        self.perform_transformation()

    # performs the training of the normalizing flows
    def setup_flow(self):

        # setting up some flow parameters. Change this to be read from a file or something like that
        n_coupling_blocks = 6
        number_of_nodes   = 256
        lr = 1e-3

        # The library we are using is zuko! 
        flow = zuko.flows.NSF(self.training_inputs.size()[1], context=self.training_conditions.size()[1], transforms=n_coupling_blocks, hidden_features=[number_of_nodes] * 3)
        flow.to(self.device)
        self.flow = flow

        self.optimizer = torch.optim.AdamW(self.flow.parameters(), lr=lr, weight_decay=1e-5)

        #defining the parameters of the lr scheduler and early stopper!
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        
        patiente_early = 11
        self.early_stopper = EarlyStopper(patience = patiente_early, min_delta=0.000)

        # Two arrays arecreated to keep track of the loss during training
        self.training_loss_array   = []
        self.validation_loss_array = [] 

    def train_the_flow(self):

        # This is a hotfix, to make all the tensors and the network have the same dtype
        self.training_inputs = self.training_inputs.type( dtype = self.training_conditions.dtype )
        self.flow = self.flow.type(   self.training_inputs.dtype )

        # Normalizing the weights to one
        self.training_weights = self.training_weights/torch.sum( self.training_weights )

        for epoch in range(15):
            for batch in range(1250):

                #making the graidients zero
                self.optimizer.zero_grad()

                # "Sampling" the batch from the array
                idxs = torch.randint(low=0, high= self.training_inputs.size()[0], size= (1024,))

                loss = self.training_weights[idxs]*(-self.flow(self.training_conditions[idxs]).log_prob( self.training_inputs[idxs]))
                loss = loss.mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0e-1)

                self.optimizer.step()

            # Switching off the gradients calculation for the validation phase
            with torch.no_grad():

                # evaluatin the validation loss
                idxs = torch.randint(low=0, high= self.validation_conditions.size()[0], size= (10024,))

                validation_loss = (1e6)*torch.tensor(self.validation_weights[idxs]).to(self.device)*(-self.flow(self.validation_conditions[idxs]).log_prob( self.validation_inputs[idxs] ))
                validation_loss = validation_loss.mean()

                # saving the losses for further plotting and monitoring!
                self.training_loss_array.append(   float(1e6*loss) )
                self.validation_loss_array.append( float(validation_loss) )

                print( 'Epoch: ', epoch , ' Training loss: ', float( loss*1e6 ) , ' Validation loss: ', float(validation_loss) )

        plot_utils.plot_loss_cruve(self.training_loss_array, self.validation_loss_array)
        # call the function that performs the final plots!
        self.evaluate_results()

    # load the trained model and perform the final plots!
    def evaluate_results(self):
        
        # Results are evaluated in the test dataset, never seen by the neural network!
        # A mask is applied so only MC events are chossen from the data space
        MaskOnlyvalidationMC       = self.validation_conditions[:,self.validation_conditions.size()[1] -1 ] == 0
        self.validation_conditions = self.validation_conditions[MaskOnlyvalidationMC]
        self.validation_inputs     = self.validation_inputs[MaskOnlyvalidationMC]
        self.validation_weights    = self.validation_weights[MaskOnlyvalidationMC]

        with torch.no_grad():

            trans      = self.flow(self.mc_test_conditions).transform
            sim_latent = trans(    self.mc_test_inputs )

            self.mc_test_conditions = torch.tensor(np.concatenate( [ self.mc_test_conditions[:,:-1].cpu() , np.ones_like( self.mc_test_conditions[:,0].cpu() ).reshape(-1,1) ], axis =1 )).to( self.device )

            trans2 = self.flow(self.mc_test_conditions).transform
            self.samples = trans2.inv( sim_latent)

            # now, lets invert the transformations
            self.invert_transformation()

            # Now I am brining the tensors back to cpu for plotting porpuses
            self.validation_inputs       = self.validation_inputs.to('cpu')
            self.data_validation_inputs  = self.data_validation_inputs .to('cpu')
            self.validation_weights      = self.validation_weights.to('cpu')
            self.samples                 = self.samples.to('cpu') 
            self.mc_test_inputs          = self.mc_test_inputs.to('cpu')
            self.mc_test_inputs          = self.mc_test_inputs.to('cpu')

            # I guess I should use the mc_validaiton tensor instead of this -> self.validation_inputs
            plot_utils.plot_distributions_for_tensors( np.array(self.data_test_inputs) , np.array(self.mc_test_inputs), np.array(self.samples), np.array(self.mc_test_weights.to('cpu')) )

        return 0


    # this function will be responsable to perform the transformations in data
    def perform_transformation(self):

        # Lets now apply the Isolation transformation into the isolation variables
        # This "self.indexes_for_iso_transform" are the indexes of the variables in the inputs= tensors where the isolation variables are stored
        # and thus, where the transformations will be performed
        self.indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14]
        self.vector_for_iso_constructors_mc   = []
        self.vector_for_iso_constructors_data = []

        # creating the constructors
        for index in self.indexes_for_iso_transform:
            self.vector_for_iso_constructors_data.append( Make_iso_continuous(self.data_training_inputs[:,index]) )
            self.vector_for_iso_constructors_mc.append( Make_iso_continuous(self.mc_training_inputs[:,index]) )

        # now really applying the transformations
        counter = 0
        for index in self.indexes_for_iso_transform:
            
            # transforming the training dataset 
            self.data_training_inputs[:,index] = self.vector_for_iso_constructors_data[counter].shift_and_sample(self.data_training_inputs[:,index])
            self.mc_training_inputs[:,index] = self.vector_for_iso_constructors_mc[counter].shift_and_sample(self.mc_training_inputs[:,index])

            # transforming the validation dataset
            self.data_validation_inputs[:,index] = self.vector_for_iso_constructors_data[counter].shift_and_sample(self.data_validation_inputs[:,index])
            self.mc_validation_inputs[:,index] = self.vector_for_iso_constructors_mc[counter].shift_and_sample(self.mc_validation_inputs[:,index])

            # for the test dataset, we only have to transform the mc part
            self.mc_test_inputs[:,index] = self.vector_for_iso_constructors_mc[counter].shift_and_sample(self.mc_test_inputs[:,index])

            counter = counter + 1

        self.mc_test_inputs = self.mc_test_inputs.to(self.device)

        # we fuse the data and mc tensors together, so they are trainied with both events
        self.training_inputs       = torch.cat([ self.data_training_inputs, self.mc_training_inputs], axis = 0).to(self.device)
        self.training_conditions   = torch.cat([ self.data_training_conditions, self.mc_training_conditions], axis = 0).to(self.device)
        self.training_weights      = torch.tensor(np.concatenate([ self.data_training_weights, self.mc_training_weights], axis = 0)).to(self.device)

        self.validation_inputs     = torch.cat([ self.data_validation_inputs, self.mc_validation_inputs], axis = 0).to(self.device)
        self.validation_conditions = torch.cat([ self.data_validation_conditions, self.mc_validation_conditions], axis = 0).to(self.device).to(self.device)
        self.validation_weights    = torch.tensor(np.concatenate([ self.data_validation_weights, self.mc_validation_weights], axis = 0)).to(self.device)

        # We now perform the standartization of the training and validation arrays
        self.input_mean_for_std = torch.mean( self.training_inputs, 0 )
        self.input_std_for_std = torch.std( self.training_inputs, 0 )
        
        # the last element of the condition tensor is a boolean, so of couse we do not transform that xD
        self.condition_mean_for_std = torch.mean( self.training_conditions[:,:-1], 0 )
        self.condition_std_for_std  = torch.std( self.training_conditions[:,:-1], 0 )
        
        # transorming the training tensors
        self.training_inputs = ( self.training_inputs - self.input_mean_for_std  )/self.input_std_for_std
        self.training_conditions[:,:-1] = ( self.training_conditions[:,:-1] - self.condition_mean_for_std )/self.condition_std_for_std

        # transforming the validaiton tensors
        self.validation_inputs = ( self.validation_inputs - self.input_mean_for_std  )/self.input_std_for_std
        self.validation_conditions[:,:-1] = ( self.validation_conditions[:,:-1] - self.condition_mean_for_std  )/self.condition_std_for_std

        # Now the test tensor
        self.mc_test_inputs = ( self.mc_test_inputs - self.input_mean_for_std  )/self.input_std_for_std
        self.mc_test_conditions[:,:-1] = ( self.mc_test_conditions[:,:-1] - self.condition_mean_for_std  )/self.condition_std_for_std

    # this function will be responsable to perform the inverse transformations in data
    def invert_transformation(self):
        
        # transorming the training tensors
        self.training_inputs = ( self.training_inputs*self.input_std_for_std + self.input_mean_for_std  )
        self.training_conditions[:,:-1] = ( self.training_conditions[:,:-1]*self.condition_std_for_std + self.condition_mean_for_std  )

        # inverse transforming the validation tensors
        self.validation_inputs = ( self.validation_inputs*self.input_std_for_std + self.input_mean_for_std  )
        self.validation_conditions[:,:-1] = ( self.validation_conditions[:,:-1]*self.condition_std_for_std - self.condition_mean_for_std  )

        # inverse transforming the test tensors
        self.mc_test_inputs = ( self.mc_test_inputs*self.input_std_for_std + self.input_mean_for_std  )
        self.mc_test_conditions[:,:-1] = ( self.mc_test_conditions[:,:-1]*self.condition_std_for_std - self.condition_mean_for_std  )

        self.samples = ( self.samples*self.input_std_for_std + self.input_mean_for_std  )

        # now inverting the isolation transformation
        # now really applying the transformations
        counter = 0
        for index in self.indexes_for_iso_transform:
            
            #self.data_validation_inputs[:,index] = self.vector_for_iso_constructors_data[counter].inverse_shift_and_sample(self.data_validation_inputs[:,index])
            #self.mc_validation_inputs[:,index]   = self.vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(self.mc_validation_inputs[:,index])
            
            # for the test dataset, we only have to transform the mc part
            self.mc_test_inputs[:,index] = self.vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(self.mc_test_inputs[:,index], processed = True)

            # now transforming the 
            self.samples[:,index] = self.vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(self.samples[:,index], processed = True)

            counter = counter + 1



    # this funcitions is responable for reading the files processed and saved by the read_data.py files
    # The tensors should already be ready for training, just plug and play!
    def read_saved_tensor(self):

        # this is defined in the read_data.py ...
        path_to_save_tensors = "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/"

        # reading the training tesnors
        self.data_training_inputs      = torch.load( path_to_save_tensors + 'data_training_inputs.pt' )
        self.data_training_conditions  = torch.load( path_to_save_tensors + 'data_training_conditions.pt')
        self.data_training_weights     = torch.load( path_to_save_tensors + 'data_training_weights.pt') 

        self.mc_training_inputs        = torch.load( path_to_save_tensors + 'mc_training_inputs.pt' )
        self.mc_training_conditions    = torch.load( path_to_save_tensors + 'mc_training_conditions.pt')
        self.mc_training_weights       = torch.load( path_to_save_tensors + 'mc_training_weights.pt') 

        # now the validation tensors
        self.data_validation_inputs     = torch.load( path_to_save_tensors + 'data_validation_inputs.pt' )
        self.data_validation_conditions = torch.load( path_to_save_tensors + 'data_validation_conditions.pt')
        self.data_validation_weights    = torch.load( path_to_save_tensors + 'data_validation_weights.pt') 

        self.mc_validation_inputs       = torch.load( path_to_save_tensors + 'mc_validation_inputs.pt' )
        self.mc_validation_conditions   = torch.load( path_to_save_tensors + 'mc_validation_conditions.pt')
        self.mc_validation_weights      = torch.load( path_to_save_tensors + 'mc_validation_weights.pt') 

        # now the test tensors
        self.data_test_inputs           = torch.load( path_to_save_tensors + 'data_test_inputs.pt' )
        self.data_test_conditions       = torch.load( path_to_save_tensors + 'data_test_conditions.pt')
        self.data_test_weights          = torch.load( path_to_save_tensors + 'data_test_weights.pt') 

        self.mc_test_inputs             = torch.tensor(torch.load( path_to_save_tensors + 'mc_test_inputs.pt' )).clone().detach()
        self.mc_test_conditions         = torch.tensor(torch.load( path_to_save_tensors + 'mc_test_conditions.pt')).clone().detach().to(self.device)
        self.mc_test_weights            = torch.tensor(torch.load( path_to_save_tensors + 'mc_test_weights.pt')).clone().detach().to(self.device) 

# this is the class responsable for the isolation variables transformation
class Make_iso_continuous:
    def __init__(self, tensor, b = False):
        
        #tensor = tensor.cpu()
        self.iso_bigger_zero  = tensor > 0 
        self.iso_equal_zero   = tensor == 0
        self.lowest_iso_value = torch.min( tensor[self.iso_bigger_zero] )
        self.shift = 0.05
        if( b ):
            self.shift = 0.5
        self.n_zero_events = torch.sum( self.iso_equal_zero )
        self.before_transform = tensor.clone().detach()
        #self.tensor_dtype = tensor.dtype()

    #Shift the continous part of the continous distribution to (self.shift), and then sample values for the discontinous part
    def shift_and_sample(self, tensor):
        #tensor = tensor.cpu()
        
        bigger_than_zero     = tensor  > 0
        tensor_zero          = tensor == 0
        self.lowest_iso_value = 0 #torch.min( tensor[ tensor_zero ] )

        #print( tensor.dtype , tensor[ bigger_than_zero ].dtype , tensor[ tensor_zero ].dtype )

        tensor[ bigger_than_zero ] = + self.shift -self.lowest_iso_value + tensor[ bigger_than_zero ]
        tensor[ tensor_zero ]    = torch.tensor(np.random.triangular( left = 0. , mode = 0, right = self.shift*0.99, size = tensor[tensor_zero].size()[0]   ), dtype = tensor[ tensor_zero ].dtype )
        tensor = torch.log(  1e-3 + tensor ) #performing the log trnasform on top of the smoothing!
        
        return tensor.to('cuda')
    
    #inverse operation of the above shift_and_sample transform
    def inverse_shift_and_sample( self,tensor , processed = False):
        #tensor = tensor.cpu()
        tensor = torch.exp( tensor ) - 1e-3

        bigger_than_shift = tensor > self.shift
        lower_than_shift  = tensor < self.shift

        tensor[ lower_than_shift  ] = 0
        tensor[ bigger_than_shift ] = - self.shift +self.lowest_iso_value + tensor[ bigger_than_shift ]
        

        #making sure the inverse operation brough exaclty the same tensor back!
        if( processed == True ):
            pass
        else:
            assert (abs(torch.sum(  self.before_transform - tensor )) < tensor.size()[0]*1e-6 )
            #added the tensor.size()[0]*1e-6 term due to possible numerical fluctioations!
        
            
        return tensor.to('cuda')