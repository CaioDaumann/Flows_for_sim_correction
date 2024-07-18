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
   
    def __init__(self, configuration, n_transforms, n_splines_bins, aux_nodes, aux_layers, max_epoch_number, initial_lr, batch_size):

        # if False, the inputs are not standardized!
        self.perform_std_transform = True

        print( 'Standartization: ', self.perform_std_transform )

        # Checking if cuda is avaliable
        print('Checking cuda avaliability: ', torch.cuda.is_available())
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.device = device

        #reading important tensors
        self.read_saved_tensor()
        self.remove_energy_raw_from_training()
        self.perform_transformation()
        
        # defining the flow hyperparametrs as menbers of the class
        self.n_transforms   = n_transforms
        self.n_splines_bins = n_splines_bins
        self.aux_nodes      = aux_nodes
        self.aux_layers     = aux_layers

        # general training hyperparameters
        self.max_epoch_number = max_epoch_number
        self.initial_lr       = initial_lr
        self.batch_size       = batch_size

        # Now, lets open a directory to store the results and models of a given configuration
        self.configuration =  configuration
        #lets create a folder with the results
        try:
            print('\nThis run dump folder: ', os.getcwd() + '/results/LHCP_results/' +self.configuration + '/')
            #already creating the folders to store the flow states and the plots
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/',  exist_ok=True)
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/saved_states/',  exist_ok=True)
        except:
            print('\nIt was not possible to open the dump folder')
            exit()

        # folder to which the code will store the results
        self.dump_folder = os.getcwd() + '/results/' +self.configuration + '/'


    # performs the training of the normalizing flows
    def setup_flow(self):


        # The library we are using is zuko! - passes = 2 for coupling blocks!
        flow = zuko.flows.NSF(self.training_inputs.size()[1], context=self.training_conditions.size()[1], bins = self.n_splines_bins ,transforms = self.n_transforms, hidden_features=[self.aux_nodes] * self.aux_layers, passes = 2)
        flow.to(self.device)
        self.flow = flow

        self.optimizer = torch.optim.AdamW(self.flow.parameters(), lr= self.initial_lr, weight_decay=1e-6)

        #defining the parameters of the lr scheduler and early stopper!
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 8)
        
        patiente_early = 12
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

        for epoch in range(999):
            for batch in range(2000):

                #making the graidients zero
                self.optimizer.zero_grad()

                # "Sampling" the batch from the array
                idxs = torch.randint(low=0, high= self.training_inputs.size()[0], size= ( self.batch_size,))

                loss = self.training_weights[idxs]*(-self.flow(self.training_conditions[idxs]).log_prob( self.training_inputs[idxs]))
                loss = loss.mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0e-1)

                self.optimizer.step()

            # End of the epoch! - calculating the validation loss and saving nescessary information!
            torch.save(self.flow.state_dict(), os.getcwd() + "/results/saved_states/epoch_"+str(epoch)+".pth")
            # Switching off the gradients calculation for the validation phase
            with torch.no_grad():

                # evaluatin the validation loss
                idxs = torch.randint(low=0, high= self.validation_conditions.size()[0], size= (10024,))

                validation_loss = (1e6)*torch.tensor(self.validation_weights[idxs]).to(self.device)*(-self.flow(self.validation_conditions[idxs]).log_prob( self.validation_inputs[idxs] ))
                validation_loss = validation_loss.mean()

                # saving the losses for further plotting and monitoring!
                self.training_loss_array.append(   float(1e6*loss) )
                self.validation_loss_array.append( float(validation_loss) )

                # updating the lr scheduler
                self.scheduler.step( validation_loss )

                print( 'Epoch: ', epoch , ' Training loss: ', float( loss*1e6 ) , ' Validation loss: ', float(validation_loss) )

                if( self.early_stopper.early_stop( float( validation_loss ) ) or epoch > self.max_epoch_number ):

                    print( 'Best epoch loss: ', np.min(  np.array( self.validation_loss_array )  ) , ' at epoch: ', np.argmin( np.array( np.array( self.validation_loss_array ) ) ) )
                    
                    # Lets select the model with the best validation loss
                    self.flow.load_state_dict(torch.load( './results//saved_states/epoch_'+str(np.argmin( np.array( np.array( self.validation_loss_array )) )) +'.pth'))
                    torch.save(self.flow.state_dict(), self.dump_folder + "/best_model_.pth")
        
                    print('\nEnd of model training! Now procedding to performance evaluation. This may take a while ... \n')

                    break

        plot_utils.plot_loss_cruve(self.training_loss_array, self.validation_loss_array, self.dump_folder)
        # call the function that performs the final plots!
        self.evaluate_results()

    # load the trained model and perform the final plots!
    def evaluate_results(self):
        
        with torch.no_grad():

            # Results are evaluated in the test dataset, never seen by the neural network!
            # A mask is applied so only MC events are chossen from the data space
            MaskOnlyvalidationMC       = self.validation_conditions[:,self.validation_conditions.size()[1] -1 ] == 0
            self.validation_conditions = self.validation_conditions[MaskOnlyvalidationMC]
            self.validation_inputs     = self.validation_inputs[MaskOnlyvalidationMC]
            self.validation_weights    = self.validation_weights[MaskOnlyvalidationMC]

            trans      = self.flow(self.mc_test_conditions).transform
            sim_latent = trans(    self.mc_test_inputs )

            self.mc_test_conditions = torch.tensor(np.concatenate( [ self.mc_test_conditions[:,:-1].cpu() , np.ones_like( self.mc_test_conditions[:,0].cpu() ).reshape(-1,1) ], axis =1 )).to( self.device )

            trans2 = self.flow(self.mc_test_conditions).transform
            self.samples = trans2.inv( sim_latent)

            # lets save the means and std used for the transformations
            np.save( os.getcwd() + '/results/' +self.configuration + '/' +  'input_means.npy', self.input_mean_for_std)
            np.save( os.getcwd() + '/results/' +self.configuration + '/' +  'input_std.npy'  , self.input_std_for_std )

            np.save( os.getcwd() + '/results/' +self.configuration + '/' +  'conditions_means.npy', self.condition_mean_for_std)
            np.save( os.getcwd() + '/results/' +self.configuration + '/'  +  'conditions_std.npy' , self.condition_std_for_std )

            # now, lets invert the transformations
            self.invert_transformation()

            # Now I am brining the tensors back to cpu for plotting porpuses
            self.validation_inputs       = self.validation_inputs.to('cpu')
            self.data_validation_inputs  = self.data_validation_inputs .to('cpu')
            self.validation_weights      = self.validation_weights.to('cpu')
            self.samples                 = self.samples.to('cpu') 
            self.mc_test_inputs          = self.mc_test_inputs.to('cpu')
            self.mc_test_inputs          = self.mc_test_inputs.to('cpu')

            self.stitch_energy_raw_from_training()

            # I guess I should use the mc_validaiton tensor instead of this -> self.validation_inputs
            plot_utils.plot_distributions_for_tensors( np.array(self.data_test_inputs) , np.array(self.mc_test_inputs), np.array(self.samples), np.array(self.mc_test_weights.to('cpu')), self.dump_folder )

            # Plotting the correlation matrices to better understand how the flows treats the correlations
            plot_utils.plot_correlation_matrix_diference_barrel(self.data_test_inputs.clone().detach().cpu(), self.data_test_conditions.clone().detach().cpu(), self.data_test_weights.clone().detach().cpu(),  self.mc_test_inputs.clone().detach().cpu(), self.mc_test_conditions.clone().detach().cpu(), self.mc_test_weights.clone().detach().cpu() , self.samples.clone().detach().cpu(),  self.dump_folder)
            plot_utils.plot_correlation_matrix_diference_endcap(self.data_test_inputs.clone().detach().cpu(), self.data_test_conditions.clone().detach().cpu(), self.data_test_weights.clone().detach().cpu(),  self.mc_test_inputs.clone().detach().cpu(), self.mc_test_conditions.clone().detach().cpu(), self.mc_test_weights.clone().detach().cpu() , self.samples.clone().detach().cpu(),  self.dump_folder)

            # Now we evaluate the run3 mvaID and check how well the distributions agree
            #(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights,path_plot)
            plot_utils.plot_mvaID_curve(np.array(self.mc_test_inputs),np.array(self.data_test_inputs),np.array(self.samples),np.array(self.mc_test_conditions.to('cpu')), np.array(self.data_test_conditions.to('cpu')),  np.array(self.mc_test_weights.to('cpu')), np.array(self.data_test_weights), self.dump_folder)
            plot_utils.plot_mvaID_curve_endcap(np.array(self.mc_test_inputs),np.array(self.data_test_inputs),np.array(self.samples),np.array(self.mc_test_conditions.to('cpu')), np.array(self.data_test_conditions.to('cpu')),  np.array(self.mc_test_weights.to('cpu')), np.array(self.data_test_weights), self.dump_folder)

        return 0


    # this function will be responsable to perform the transformations in data
    def perform_transformation(self):

        # Lets now apply the Isolation transformation into the isolation variables
        # This "self.indexes_for_iso_transform" are the indexes of the variables in the inputs= tensors where the isolation variables are stored
        # and thus, where the transformations will be performed
        self.indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14] #[7,8,9,10,11,12,13,14,15] #[6,7,8,9,10,11,12,13,14] #this has to be changed once I take the energy raw out of the inputs
        self.vector_for_iso_constructors_mc   = []
        self.vector_for_iso_constructors_data = []

        # creating the constructors
        for index in self.indexes_for_iso_transform:
            
            # since hoe has very low values, the shift value (value until traingular events are sampled) must be diferent here
            if( index == 6 ):
                self.vector_for_iso_constructors_data.append( Make_iso_continuous(self.data_training_inputs[:,index], self.device , b = 0.001) )
                self.vector_for_iso_constructors_mc.append( Make_iso_continuous(self.mc_training_inputs[:,index], self.device  , b= 0.001) )
            else:
                self.vector_for_iso_constructors_data.append( Make_iso_continuous(self.data_training_inputs[:,index], self.device ) )
                self.vector_for_iso_constructors_mc.append( Make_iso_continuous(self.mc_training_inputs[:,index], self.device ) )

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

        # We now perform the standartization of the training and validation arrays - lets use only mc to calculate the means and stuff ...
        self.input_mean_for_std = torch.mean( self.training_inputs[  self.training_conditions[:,  self.training_conditions.size()[1] -1 ] == 0   ], 0 )
        self.input_std_for_std  = torch.std( self.training_inputs[  self.training_conditions[:,  self.training_conditions.size()[1] -1 ] == 0   ], 0 )
        
        # the last element of the condition tensor is a boolean, so of couse we do not transform that xD
        self.condition_mean_for_std = torch.mean( self.training_conditions[:,:-1][  self.training_conditions[:,  self.training_conditions.size()[1] -1 ] == 0   ], 0 )
        self.condition_std_for_std  = torch.std( self.training_conditions[:,:-1][  self.training_conditions[:,  self.training_conditions.size()[1] -1 ] == 0   ], 0 )
        
        """
        print( self.input_mean_for_std )
        print( self.input_std_for_std )
        print( self.condition_mean_for_std )
        print( self.condition_std_for_std )
        exit()
        """
    
        # transorming the training tensors
        if( self.perform_std_transform ):
            self.training_inputs = ( self.training_inputs - self.input_mean_for_std  )/self.input_std_for_std
            self.training_conditions[:,:-1] = ( self.training_conditions[:,:-1] - self.condition_mean_for_std )/self.condition_std_for_std

            # transforming the validaiton tensors
            self.validation_inputs = ( self.validation_inputs - self.input_mean_for_std  )/self.input_std_for_std
            self.validation_conditions[:,:-1] = ( self.validation_conditions[:,:-1] - self.condition_mean_for_std  )/self.condition_std_for_std

            # Now the test tensor
            self.mc_test_inputs = ( self.mc_test_inputs - self.input_mean_for_std  )/self.input_std_for_std
            self.mc_test_conditions[:,:-1] = ( self.mc_test_conditions[:,:-1] - self.condition_mean_for_std  )/self.condition_std_for_std

        # Lets now plot the distirbutions after the transformations
        plot_utils.plot_distributions_after_transformations(self.training_inputs.clone().detach().cpu(), self.training_conditions.clone().detach().cpu(), self.training_weights.clone().detach().cpu())

    # this function will be responsable to perform the inverse transformations in data
    def invert_transformation(self):
        
        # transorming the training tensors
        if( self.perform_std_transform ):
            self.training_inputs = ( self.training_inputs*self.input_std_for_std + self.input_mean_for_std  )
            self.training_conditions[:,:-1] = ( self.training_conditions[:,:-1]*self.condition_std_for_std + self.condition_mean_for_std  )

            # inverse transforming the validation tensors
            self.validation_inputs = ( self.validation_inputs*self.input_std_for_std + self.input_mean_for_std  )
            self.validation_conditions[:,:-1] = ( self.validation_conditions[:,:-1]*self.condition_std_for_std - self.condition_mean_for_std  )

            # inverse transforming the test tensors
            self.mc_test_inputs = ( self.mc_test_inputs*self.input_std_for_std + self.input_mean_for_std  )
            self.mc_test_conditions[:,:-1] = ( self.mc_test_conditions[:,:-1]*self.condition_std_for_std + self.condition_mean_for_std  )

            self.samples = ( self.samples*self.input_std_for_std + self.input_mean_for_std  )

        # Now inverting the isolation transformation
        counter = 0
        for index in self.indexes_for_iso_transform:
            
            # for the test dataset, we only have to transform the mc part
            self.mc_test_inputs[:,index] = self.vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(self.mc_test_inputs[:,index], processed = True)

            # now transforming the 
            self.samples[:,index] = self.vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(self.samples[:,index], processed = True)

            counter = counter + 1

    # energy raw should not be correct, So i am removing it from the inputs here! it is always the first entry in the tensor
    def remove_energy_raw_from_training(self):

        self.data_training_inputs = self.data_training_inputs[:,1:]
        self.mc_training_inputs   = self.mc_training_inputs[:,1:]
        
        self.data_validation_inputs = self.data_validation_inputs[:,1:]
        self.mc_validation_inputs   = self.mc_validation_inputs[:,1:]
        
        # but we still need it to reevaluate the mvaID, so I we need to keep it!
        self.mc_test_inputs_energy_raw = self.mc_test_inputs[:,0]
        self.mc_test_inputs = self.mc_test_inputs[:,1:]

    def stitch_energy_raw_from_training(self):
        # re-adding teh energy raw for the mvaID calculation!
        self.mc_test_inputs = torch.cat( [ self.mc_test_inputs_energy_raw.view(-1,1) , self.mc_test_inputs   ], axis = 1 )
        self.samples        = torch.cat( [ self.mc_test_inputs_energy_raw.view(-1,1) , self.samples  ], axis = 1 )

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
    def __init__(self, tensor, device, b = False):
        
        self.device = device

        #tensor = tensor.cpu()
        self.iso_bigger_zero  = tensor > 0 
        self.iso_equal_zero   = tensor == 0
        #self.lowest_iso_value = torch.min( tensor[self.iso_bigger_zero] )
        self.shift = 0.05 #era 0.05
        if( b ):
            self.shift = b
        self.n_zero_events = torch.sum( self.iso_equal_zero )
        self.before_transform = tensor.clone().detach()
        #self.tensor_dtype = tensor.dtype()

    #Shift the continous part of the continous distribution to (self.shift), and then sample values for the discontinous part
    def shift_and_sample(self, tensor):
        
        # defining two masks to keep track of the events in the 0 peak and at the continous tails
        bigger_than_zero      = tensor  > 0
        tensor_zero           = tensor == 0
        self.lowest_iso_value = 0.0 #torch.min( tensor[ bigger_than_zero ] )

        tensor[ bigger_than_zero ] = tensor[ bigger_than_zero ] + self.shift -self.lowest_iso_value 
        tensor[ tensor_zero ]      = torch.tensor(np.random.triangular( left = 0. , mode = 0, right = self.shift*0.99, size = tensor[tensor_zero].size()[0]   ), dtype = tensor[ tensor_zero ].dtype )
        
        # now a log trasform is applied on top of the smoothing to stretch the events in the 0 traingular and "kill" the iso tails
        tensor = torch.log(  1e-3 + tensor ) 
        
        return tensor.to(self.device)
    
    #inverse operation of the above shift_and_sample transform
    def inverse_shift_and_sample( self,tensor , processed = False):
        #tensor = tensor.cpu()
        tensor = torch.exp( tensor ) - 1e-3

        bigger_than_shift = tensor > self.shift
        lower_than_shift  = tensor < self.shift

        tensor[ lower_than_shift  ] = 0
        tensor[ bigger_than_shift ] = tensor[ bigger_than_shift ] - self.shift + self.lowest_iso_value 
        

        #making sure the inverse operation brough exaclty the same tensor back!
        if( processed == True ):
            pass
        else:
            assert (abs(torch.sum(  self.before_transform - tensor )) < tensor.size()[0]*1e-6 )
            #added the tensor.size()[0]*1e-6 term due to possible numerical fluctioations!
              
        return tensor.to(self.device)