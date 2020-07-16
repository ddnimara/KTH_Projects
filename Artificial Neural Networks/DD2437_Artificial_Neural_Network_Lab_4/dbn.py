from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size,use_two_layers=False):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """
        use_two_layers=False
        self.use_two_layers=use_two_layers
        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 800
        
        self.n_gibbs_wakesleep = 20

        self.print_period = 2000

        self.mean_of_prob = None
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        first_layer_h = self.rbm_stack['vis--hid'].get_h_given_v_dir(true_img)[0]
        if(self.use_two_layers==False):

            pen_layer_h =  self.rbm_stack['hid--pen'].get_h_given_v_dir(first_layer_h)[0]
        else:
            pen_layer_h=first_layer_h

        v_0 = np.concatenate((pen_layer_h,lbl),axis=1)
        print('I am about to enter the gibbs')
        for _ in range(self.n_gibbs_recog):
            print('currently in iteration ', _)
            h_0 = self.rbm_stack['pen+lbl--top'].get_h_given_v(v_0)[1]
            v_0 = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_0)[0]

        print("I just exited the gibbs")
        predicted_lbl = v_0[:,-self.rbm_stack['pen+lbl--top'].n_labels:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name,train_imgs):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).


        #vis = np.random.choice([0,1],size= self.sizes["vis"]).reshape(1,self.sizes['vis'])
        #vis = np.ones(self.sizes["vis"]).reshape(1,self.sizes['vis'])
        #vis = sigmoid(self.rbm_stack['vis--hid'].bias_v).reshape(1,-1)
        #vis = train_imgs[0].reshape(1,-1)
        #plt.imshow(vis.reshape(self.image_size))
        #plt.show()
        #vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)[0]
        #vis = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)[0]
        vis = self.mean_of_prob.reshape(1,-1)
        #vis = np.random.choice([0, 1], size=self.sizes["pen"]).reshape(1, self.sizes['pen'])
        vis = np.concatenate((vis, lbl), axis=1)

        for _ in range(self.n_gibbs_gener):
            h_0 = self.rbm_stack['pen+lbl--top'].get_h_given_v(vis)[1]
            vis,vis_states = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_0)
            vis[:,-10:] =lbl
            if(self.use_two_layers==False):
                vis_mid = self.rbm_stack['hid--pen'].get_v_given_h_dir(vis[:,:-10])[0]
            else:
                vis_mid = vis_states[:,:-10]
            vis_bot = self.rbm_stack['vis--hid'].get_v_given_h_dir(vis_mid)[1]
            records.append( [ ax.imshow(vis_bot.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            


            if(self.use_two_layers==False):
                self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
                self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")

            self.mean_of_prob = np.load("mean_probabilities_top.npy", allow_pickle = True)

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily

            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            self.rbm_stack['vis--hid'].cd1(vis_trainset)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()            
            """ 
            CD-1 training for hid--pen 
            """

            train_data_for_intermediate_layer = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_trainset)[0]
            if (self.use_two_layers == False):
                self.rbm_stack['hid--pen'].cd1(train_data_for_intermediate_layer)
                self.savetofile_rbm(loc="trained_rbm",name="hid--pen")


                self.rbm_stack["hid--pen"].untwine_weights()
            print("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            if(self.use_two_layers==False):
                probs = self.rbm_stack['hid--pen'].get_h_given_v_dir(train_data_for_intermediate_layer)[0]
            else:
                probs = train_data_for_intermediate_layer
            train_data_for_last_layer = np.concatenate((probs,lbl_trainset),axis=1)
            self.rbm_stack['pen+lbl--top'].cd1(train_data_for_last_layer)

            self.mean_of_prob = np.mean(probs, axis = 0)
            
            np.save('mean_probabilities_top.npy', self.mean_of_prob)

            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            if(self.use_two_layers==False):
                self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]
            total_index = np.arange(self.n_samples)
            total_index = np.random.permutation(total_index)
            for epochs in range(1):
                for it in range(n_iterations):
                    print("Iteration :", it)

                    indeces = total_index[it*self.batch_size:(it+1)*self.batch_size]
                    v_0_batch = vis_trainset[indeces]
                    lbl_batch = lbl_trainset[indeces]

                    # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                    lbl_vis_hid_wake_prob,lbl_vis_hid_wake_activ = self.rbm_stack["vis--hid"].get_h_given_v_dir(v_0_batch)
                    if(self.use_two_layers==False):
                        lbl_hid_pen_wake_prob,lbl_hid_pen_wake_activ = self.rbm_stack["hid--pen"].get_h_given_v_dir(lbl_vis_hid_wake_activ)
                    else:
                        lbl_hid_pen_wake_prob, lbl_hid_pen_wake_activ = lbl_vis_hid_wake_prob,lbl_vis_hid_wake_activ
                    lbl_hid_pen_wake_activ = np.concatenate((lbl_hid_pen_wake_activ, lbl_batch), axis = 1)
                    lbl_hid_pen_wake_prob = np.concatenate((lbl_hid_pen_wake_prob, lbl_batch), axis = 1)

                    # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                    for _ in range(self.n_gibbs_wakesleep):
                        neg_lbl_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(lbl_hid_pen_wake_activ)[1]
                        lbl_hid_pen_wake_prob_gibbs, lbl_hid_pen_wake_activ = self.rbm_stack['pen+lbl--top'].get_v_given_h(neg_lbl_top)
                        lbl_hid_pen_wake_activ[:,-lbl_batch.shape[1]:] = lbl_batch

                    # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                    lbl_top_pen_sleep_prob,  lbl_top_pen_sleep_activ= self.rbm_stack["pen+lbl--top"].get_v_given_h(neg_lbl_top)
                    if(self.use_two_layers==False):

                        lbl_pen_hid_sleep_prob,lbl_pen_hid_sleep = self.rbm_stack["hid--pen"].get_v_given_h_dir(lbl_top_pen_sleep_activ[:,:-lbl_batch.shape[1]])
                        lbl_vis_hid_sleep_prob, lbl_vis_hid_sleep = self.rbm_stack["vis--hid"].get_v_given_h_dir(
                            lbl_pen_hid_sleep)
                    else:
                        lbl_pen_hid_sleep_prob, lbl_pen_hid_sleep = lbl_top_pen_sleep_prob,  lbl_top_pen_sleep_activ
                        lbl_vis_hid_sleep_prob,lbl_vis_hid_sleep = self.rbm_stack["vis--hid"].get_v_given_h_dir(lbl_pen_hid_sleep[:,:-lbl_batch.shape[1]])

                    # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                    # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.
                    # Wake Phase

                    self.rbm_stack["vis--hid"].update_generate_params(lbl_vis_hid_wake_activ, v_0_batch, lbl_vis_hid_sleep_prob)
                    if (self.use_two_layers == False):
                        self.rbm_stack["hid--pen"].update_generate_params(lbl_hid_pen_wake_activ[:,:-lbl_batch.shape[1]], lbl_vis_hid_wake_activ, lbl_pen_hid_sleep_prob)

                    # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.
                    h_0 = self.rbm_stack['pen+lbl--top'].get_h_given_v(lbl_hid_pen_wake_prob)[1]
                    h_k = self.rbm_stack['pen+lbl--top'].get_h_given_v(lbl_hid_pen_wake_prob_gibbs)[0]
                    self.rbm_stack["pen+lbl--top"].update_params(lbl_hid_pen_wake_prob, h_0, lbl_hid_pen_wake_prob_gibbs, h_k)

                    # [TODO TASK 4.3] update recognition parameters : here you will only use 'update_recognize_params' method from rbm class.
                    # Sleep Phase
                    #self.rbm_stack["pen+lbl--top"].cdk(visible_trainset=lbl_top_pen_sleep_prob,n_iterations=20)#cd1(visible_trainset = lbl_top_pen_sleep_prob, n_iterations =20, epochs = 1, method = "wake-sleep")
                    if(self.use_two_layers==False):
                        self.rbm_stack["hid--pen"].update_recognize_params(lbl_pen_hid_sleep, lbl_top_pen_sleep_activ[:,:-lbl_batch.shape[1]], lbl_hid_pen_wake_prob[:,:-lbl_batch.shape[1]])
                        self.rbm_stack["vis--hid"].update_recognize_params(lbl_vis_hid_sleep, lbl_pen_hid_sleep, lbl_vis_hid_wake_prob)
                    else:
                        self.rbm_stack["vis--hid"].update_recognize_params(lbl_vis_hid_sleep,
                                                                           lbl_pen_hid_sleep[:, :-lbl_batch.shape[1]],
                                                                           lbl_vis_hid_wake_prob)

                    if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            if(self.use_two_layers==False):
                self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
