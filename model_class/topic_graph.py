import numpy as np
 
class TopicGraph:
    """
    class for storing model parameters and model learning
    """
    def __init__(self, D, W, level_topic_counts, ndw, params):
        self.W = W
        self.D = D
        self.ndw = ndw
        self.params = params
        self.level_topic_counts = level_topic_counts
        self.L = len(level_topic_counts)
        self.i = 0
        

    def initialize(self):
        if self.L == 0:
            print "Count of levels is 0, returning"       
            return            
        self.phis = []
        self.psis = []
        for l in np.arange(self.L-1):
            phi = np.random.rand(self.W, self.level_topic_counts[l])
            phi /= phi.sum(axis=0)[np.newaxis, :]
            self.phis.append(phi)
            psi = np.random.rand(self.level_topic_counts[l], \
                                 self.level_topic_counts[l+1])
            psi /= psi.sum(axis=0)[np.newaxis, :]
            self.psis.append(psi)
        phi = np.random.rand(self.W, self.level_topic_counts[-1])
        phi /= phi.sum(axis=0)[np.newaxis, :]
        self.phis.append(phi)
        self.theta = np.random.rand(self.level_topic_counts[-1], self.D)
        self.theta /= self.theta.sum(axis=0)[np.newaxis, :]
        self.eta = np.random.rand(self.L, self.D)
        self.eta /= self.eta.sum(axis=0)[np.newaxis, :]
        
    def construct(self, it=25):
        LL = []
        for i in range(it):
            self.i += 1
            step = 1 / float(self.i+1)
            
            #compute normalization constants
            z_wd = self.phis[-1].dot(self.theta) *\
                   self.eta[-1, :][np.newaxis, :]
            for l in np.arange(self.L-2, -1, -1):
               addit = self.phis[l]
               for ll in np.arange(l, self.L-1):
                    addit = addit.dot(self.psis[ll])
               addit = addit.dot(self.theta)
               z_wd += addit * self.eta[l, :][np.newaxis, :]
            
            L = 0
            
            #Estep
            phi_grads = [np.zeros(phi.shape) for phi in self.phis]
            psi_grads = [np.zeros(psi.shape) for psi in self.psis]
            theta_grad = np.zeros(self.theta.shape)
            eta_grad = np.zeros(self.eta.shape)
            
            
            for d in self.ndw:
                for w in self.ndw[d]:
                    L += self.ndw[d][w] * np.log(z_wd[w, d])
                    theta_col = self.theta[:, d][:, np.newaxis]
                    aggreg_thetas = {}
                    aggreg_thetas[self.L-1] = theta_col
                    for l in range(self.L-2, -1, -1):
                        theta_col = self.psis[l].dot(theta_col)
                        aggreg_thetas[l] = theta_col
                    aggreg_phi_row = self.phis[0][w, :]
                    for l in range(self.L):
                        phi_grads[l][w, :] += aggreg_thetas[l].ravel() \
                                                     * self.eta[l, d] * self.ndw[d][w] / z_wd[w, d]
                        if l < self.L-1:
                            psi_grads[l] += aggreg_phi_row[:, np.newaxis] * aggreg_thetas[l+1].T \
                                                     * self.eta[l, d] * self.ndw[d][w] / z_wd[w, d]
                            aggreg_phi_row = aggreg_phi_row.dot(self.psis[l]) + self.phis[l+1][w, :]
                        eta_grad[l, d] += self.phis[l][w, :].dot(aggreg_thetas[l].ravel()) \
                                          * self.ndw[d][w] / z_wd[w, d]
                    theta_grad[:, d] += aggreg_phi_row * self.ndw[d][w] / z_wd[w, d]
                
            #update theta
            """
            theta_grad /= theta_grad.sum(axis=0)[np.newaxis, :]
            self.theta *= 1 - step
            self.theta += step * theta_grad
            """
            self.theta *= theta_grad
            self.theta /= self.theta.sum(axis=0)[np.newaxis, :] + 1e-30
            
            #update psis
            for l in np.arange(self.L):
                if l < self.L-1:
                    """
                    psi_grads[l] /= psi_grads[l].sum(axis=0)[np.newaxis, :]
                    psi_grads[l] *= 1 - step
                    self.psis[l] += step * psi_grads[l]   
                    """
                    self.psis[l] *= psi_grads[l]
                    self.psis[l] /= self.psis[l].sum(axis=0)[np.newaxis, :] + 1e-30
                
                #update phis
                """
                phi_grads[l] /= phi_grads[l].sum(axis=0)[np.newaxis, :]
                self.phis[l] *= 1 - step
                self.phis[l] += step * phi_grads[l]
                """
                self.phis[l] *= phi_grads[l]
                self.phis[l] /= self.phis[l].sum(axis=0)[np.newaxis, :] +1e-30
            #update eta
            self.eta *= eta_grad
            self.eta /= self.eta.sum(axis=0)[np.newaxis, :] + 1e-30    
            
            LL.append(L)
        return LL
            
    def regularize(self, reg_phi=None, reg_psi=None, reg_theta=None, return_scores=False):
        if len(self.phis) == 0:
            print "There are 0 levels"
            return
        if return_scores:
            scores = {}
        if reg_phi is not None or return_scores:  #reg phi or compute scores
            # collect global phi for all levels
            common_phi = np.zeros((self.phis[0].shape[0], 0))
            theta = self.theta
            for phi_idx in range(self.L-1, -1, -1):
                p_s = (theta*(self.eta[phi_idx, :]*self.p_d)[np.newaxis, :]).sum(axis=1)
                common_phi = np.hstack((common_phi, self.phis[phi_idx]*p_s[np.newaxis, :]))
                if phi_idx > 0:
                    theta = self.psis[phi_idx-1].dot(theta)
            topics_count = common_phi.shape[1]
            common_phi /= common_phi.sum(axis=1)[:, np.newaxis] + 1e-100
            if reg_phi is not None:
                common_phi **= reg_phi
                common_phi /= common_phi.sum(axis=1)[:, np.newaxis] + 1e-100
            
            cur_idx = 0
            purity = 0
            for l in range(self.L-1, -1, -1):
                if reg_phi is not None:
                    self.phis[l][:, :] = common_phi[:, cur_idx:cur_idx+self.phis[l].shape[1]]\
                                           * self.p_w[:, np.newaxis]
                    self.phis[l] /= self.phis[l].sum(axis=0)[np.newaxis, :] + 1e-100
                if return_scores:
                    purity += (self.phis[l][common_phi[:, cur_idx:cur_idx+self.phis[l].shape[1]]>0.25]).sum()
                cur_idx += self.phis[l].shape[1]
            if return_scores:
                scores["purity"] = purity / topics_count
                scores["phi_sparsity"] = (common_phi > 0).sum() / float(common_phi.shape[0]*common_phi.shape[1])
            
        if reg_psi is not None or return_scores:
            psi_sparsity = 0
            sum_shape = 0
            for psi in self.psis:  # reg psi or compute score
                if reg_psi is not None:
                    psi **= reg_psi
                    psi /= psi.sum(axis=0)[np.newaxis, :] + 1e-100
                if return_scores:                    
                    psi_sparsity += (psi > 0).sum()
                    sum_shape += psi.shape[0] * psi.shape[1]
        if reg_theta is not None:  # reg theta or compute score
             self.theta **= reg_theta
             self.theta /= self.theta.sum(axis=0)[np.newaxis, :] + 1e-100
        if return_scores:  
            scores["psi_sparsity"] = psi_sparsity / float(sum_shape)
            scores["theta_sparsity"] = (self.theta > 0).sum() / float(self.theta.shape[0]*self.theta.shape[1])
        if return_scores:
            return scores        
        
        
        
        
        
        