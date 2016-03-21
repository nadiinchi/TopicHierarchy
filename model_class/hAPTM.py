import numpy as np
import pickle

import topic_graph

#reload(hparams)
reload(topic_graph)

class hAPTM():
    """
    a class to build hierarchy
    usage:
    
    hier = hAPTM()    
    """
    def __init__(self):
        self.graph = None
        self.params = {}
        self.dictionary = {}
        
        
    def parseFromTxt(self, txtfile):
        """
        parse collection fron txt. 
        
        Collection must be stored in uci bag-of-words format.
        params:
        txtfile --- a full path to the file with collection.    
        """
        with open(txtfile, "r") as f:
            self.D = int(f.readline().replace('\n', ''))
            self.W = int(f.readline().replace('\n', ''))
            self.NNZ = int(f.readline().replace('\n', ''))
            self.lens = np.zeros(self.D)
            self.p_w = np.zeros(self.W)
            self.ndw = {d:{} for d in range(self.D)}
            for line in f:
                 d, w, n = line.replace('\n', '').split(' ')
                 self.ndw[int(d)][int(w)] = int(n)
                 self.lens[int(d)] += int(n)
                 self.p_w[int(w)] += int(n)
            self.lens /= self.lens.sum()
            self.p_w /= self.p_w.sum()
            
            
    def parseDictionaryFromTxt(self, txtfile, idxs_in_file=True):
        """
        parse dictionary that matches word_id with word.
        
        params:
        txtfile --- a full path to the file with dictionary 
        idxs_in_file --- if True, file is considered to contain pairs ``word_id word'' per line,
                         otherwise of words per line (index is assumed to be line number).
        
        """
        with open(txtfile,"r") as file:
            for line in file:
                if idxs_in_file:
                    pair = list(line.strip().split(' '))
                    self.dictionary[int(pair[0])] = pair[1]
                else:
                    self.dictionary[len(self.dictionary)] = line[:-1]
                     
                     
                     
    def reset(self, level_topic_counts, params={}):
        """
        specify hierarchy configuration and initialize it

        params:
        level_topic_counts --- list of topic counts per hierarchy level
        params --- regularization coeffitients

        Attention! This method overwrites heirarchy, previous model will be deleted!

        """
        self.graph = topic_graph.TopicGraph(self.D, self.W, level_topic_counts, \
                                          self.ndw, params)
        self.graph.p_d = self.lens
        self.graph.p_w = self.p_w
        self.graph.initialize()
        
         
    def construct(self, it=25):
        """
        run optimization algorithm to fit model parameters
        
        params:
        it --- number of iterations to be performed
        
        Returns list of log_likelihood after each iteration
        
        Optimization will be continued (not restarted) with further method calls.
        """
        return self.graph.construct(it)
         
         
    def regularize(self, reg_phi=None, reg_psi=None, reg_theta=None, return_scores=False):
        """
        make one regularization step
        
        params:
        reg_phi --- exponent for phi matrices regularization, double, 
                    None means no regularization
        reg_psi --- exponent for psi matrices regularization, double, 
                    None means no regularization
        reg_theta --- exponent for theta regularization, double, 
                    None means no regularization
        return_scores --- if True, return metrics computed after regularization, 
                          otherwise return None
        """
        return self.graph.regularize(reg_phi, reg_psi, reg_theta, return_scores)
         
         
    def add_levels(self, new_level_topics_count):
        """
        add new levels with specified topic counts. Keep phis and psis, destruct old theta and eta
        
        new_level_topics_count --- list of topic counts per NEW hierarchy level
        """
        self.graph.add_levels(new_level_topics_count)
         

    def printLevel(self, level, matrices=set(), phi_threshold=100):
        """
        print topic snippets
        
        params:
            level --- level index to be printed, integer
            matrices --- what snippets to print, string:
                         "phi_top" --- top words for each topic 
                         "phi" --- part of phi matrix 
                         "psi" --- psi matrix 
                         "theta" --- part of theta.T 
            phi_threshold --- for "phi_top" snippet: count of top words to be printed 
                              for "phi" snippet: count of phi lines to be printed
        
        
        """
        if level < 0 or level > len(self.graph.phis):
            print "Incorrect level index"
            return
        np.set_printoptions(edgeitems=3,infstr='inf',\
        linewidth=75, nanstr='nan', precision=8,\
        suppress=False, threshold=1000, formatter=None)
        if "phi" in matrices:
             print "Phi:"
             for i in range(min(self.graph.phis[level].shape[0], phi_threshold)):
                 if i in self.dictionary:
                     word = self.dictionary[i]
                 else:
                     word = ""
                 print "%15s"%word+' '+str(self.graph.phis[level][i])
        if "phi_top" in matrices:
             print "Top phi:"
             sorted_idxs = np.argsort(self.graph.phis[level], axis=0)
             for i in range(sorted_idxs.shape[1]):
                 print "Topic", str(i), ", ".join\
                       ([self.dictionary[j].lower() for j in sorted_idxs[:-phi_threshold:-1, i]])
        if "psi" in matrices:
             print "Psi:"
             print self.graph.psis[level]
        if "theta" in matrices:
             print "Theta.T:"
             print self.theta.T
             
             
    def save(self, filename):
        """
        dump model on disk
        
        params:
        filename --- full path of file to be created.
        
        Model is dumped using pickle
        """
        with open(filename, "w") as file:
            pickle.dump(self, file)
            

    def convertNdw(self):
        """
        returns dense np-array representing ndw matrix (D x W)
        """
        ndw = np.zeros((self.W, self.D))
        for d in self.ndw:
            for w in self.ndw[d]:
                ndw[w, d] = self.ndw[d][w]
        return ndw
                
         
         
