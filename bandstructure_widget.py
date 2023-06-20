
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import TextBox,Button
#import pythtb
from nonorthTB import TBModel
import copy

rcParams.update({'font.size': 10})

class Bandstructure(object):
    def __init__(self,wannierDir,wannierTag, numWannierOrbs=None,kpath = None, k_label = None):
        #super(Bandstructure, self).__init__(wannierDir, wannierTag, numWannierOrbs, kpath, k_label)
        self.wannierDir=wannierDir
        self.wannierTag=wannierTag
        self.numOrbs = int(numWannierOrbs)
        self.kptFile = wannierDir+"/"+wannierTag+'_band.kpt'
        self.datFile = wannierDir+"/"+wannierTag+'_band.dat'
        win_file = wannierDir+"/"+wannierTag+'.win'
        self.axBS = None

        #get info about structure
        filedata = open(win_file)
        filelines = filedata.readlines(1000)
        for line in range(100):
            if "begin unit_cell_cart" in filelines[line]:
                conv_start = line + 1
            if "begin atoms_cart" in filelines[line]:
                atom_coord_start = line + 1
            if "end atoms_cart" in filelines[line]:
                atom_coord_end = line
                break
        # set conventional cell vectors
        self._a1 = np.array([float(i) for i in filelines[conv_start].strip().split()])
        self._a2 = np.array([float(i) for i in filelines[conv_start + 1].strip().split()])
        self._a3 = np.array([float(i) for i in filelines[conv_start + 2].strip().split()])

        # set position and name of atoms
        self.numAtoms = int(atom_coord_end-atom_coord_start)
        self.atom_cartpos = np.zeros((self.numAtoms,3))
        self.atom_primpos = np.zeros((self.numAtoms,3))
        self.elements = np.empty((self.numAtoms),dtype=str)
        for atom in range(self.numAtoms):
            index = atom_coord_start + atom
            element = filelines[index].strip().split()[0]
            position = np.array([float(i) for i in filelines[index].strip().split()[1:4]])
            self.atom_cartpos[atom] = position
            self.atom_primpos[atom] = _cart_to_red((self._a1,self._a2,self._a3),[position])[0]
            self.elements[atom] = element
        print("atom prim coords:",self.atom_primpos)

        #files = pythtb.w90(wannierDir,wannierTag)
        if kpath == None:
            kpath= [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.5]]
        if k_label == None:
            k_label = [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$']

        #model = files.model(min_hopping_norm=0.00001,max_distance=4,ignorable_imaginary_part=0.00001)
        (k_vec, k_dist, k_node) = self.k_path(kpath,75)# model.k_path(kpath,75)
        #(int_evals, int_evecs) = model.solve_all(k_vec, eig_vectors=True)
        #print("shape:",int_evecs.shape)

        # my TB model
        latticev = [self._a1,self._a2,self._a3]
        atoms = self.elements
        atom_coords = self.atom_primpos
        self.newModel = TBModel(wannierDir + "/", latticev, atoms, atom_coords, orbs_orth=True,min_hopping_dist=4.)
        file = "wannier90_hr.dat"
        self.newModel.read_TBparams(file)
        self._hoppings = copy.deepcopy(self.newModel.TB_params)
        self.orb_redcoords = self.newModel.orb_redcoords

        int_evals = []
        int_evecs = []
        for kpt in k_vec:
            (evals, evecs) = self.newModel.get_ham(kpt)
            int_evals.append(evals)
            int_evecs.append(evecs)
        # end new

        self.evals = np.array(int_evals).T  #[n,k]
        self.evecs = np.array(int_evecs).transpose((2,0,1))  #[n,k,orb]
        #print("other shapes",self.evals.shape,self.evecs.shape)
        self.k_vec = k_vec
        self.k_dist = k_dist
        self.k_node = k_node
        self.k_label = k_label
        print('intialized Bandstructure')

    def plotBS(self, ax=None, selectedDot=None, plotnew=None):
        """
        :param ax (matplotlib.pyplot axis): axis to save the bandstructure to, otherwise generate new axis
        :param selectedDot (1D integer array): gives kpoint and band index of the dot selected to make green circle
                                eg: [3,4]
        :return: matplotlib.pyplot axis with bandstructure plotted
        """
        fermi_energy = 5.619735  # best si: 7.089 #used to set the valence band maximum to zero
        #print("tick info:", self.k_node, self.k_label)

        #initialize figure and axis
        fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(111)
        else:
            ax = ax

        #plot the bands, i loops over each band
        for i in range(self.evals.shape[0]):
            ax.plot(self.k_dist, self.evals[i] - fermi_energy, c='gray', picker=True, label=str(i), linewidth=3)
            if plotnew == True:
                ax.plot(self.k_dist, self.new_evals[i] - fermi_energy, c='red', picker=True, label=str(i), linewidth=3)
            #k_dist and evals[i] are 1D arrays of equal length


        #plot the vertical lines for high symmetry points
        for n in range(len(self.k_node)):
            ax.axvline(x=self.k_node[n], linewidth=0.5, color='k')

        #make the selected dot green
        if selectedDot != None:
            ax.plot(self.k_dist[selectedDot[0]], self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy, 'o',
                    c='limegreen', picker=True, label=str(i), markersize=10)

        #make plot look nice
        ax.set_xlabel("Path in k-space")  # ,fontsize=20)
        ax.set_ylabel("Energy (eV)")  # ,fontsize=20)
        ax.set_xlim(self.k_dist[0], self.k_dist[-1])
        ax.set_xticks(self.k_node)
        ax.set_xticklabels(self.k_label)  # ,fontsize=20)
        ax.set_ylim(-13, 10)
        #fig.tight_layout()
        plt.close()
        #plt.show()
        return ax

    def get_significant_bonds(self, band, kpoint):
        value = 0
        # define diff_evals

        #print(band, kpoint)
        diff_evals_vals = []
        diff_evals_keys = []
        orig_tb_params = []
        orbitals = np.arange(0,8)
        eigen_var = copy.deepcopy(self.evecs[band,kpoint,:])
        sig_orbitals = orbitals[abs(eigen_var)>0.1]
        #print("Sig_orbitals", sig_orbitals)
        num_trans = self.newModel.num_each_dir
        numT = int(num_trans*2+1)

        #do above the smart way that is much better
        scaledTBparams = self.newModel.get_ham(self.k_vec[kpoint],scaledTBparams=True)
        TBparams = copy.deepcopy(self._hoppings).flatten()

        energyCont = np.zeros((self.numOrbs, self.numOrbs,numT, numT, numT), dtype=np.complex_)
        for orb1 in range(self.numOrbs):
            for orb2 in range(self.numOrbs):
                energyCont[orb1,orb2] = scaledTBparams[orb1,orb2]*np.conj(eigen_var[orb1])*eigen_var[orb2]
        energyCont = energyCont.flatten()
        orb1,orb2,trans1,trans2,trans3 = np.mgrid[0:self.numOrbs,0:self.numOrbs,-num_trans:num_trans+1,-num_trans:num_trans+1,-num_trans:num_trans+1]
        (orb1, orb2, trans1, trans2, trans3) = (orb1.flatten(), orb2.flatten(), trans1.flatten(), trans2.flatten(), trans3.flatten())
        large_TB = np.abs(energyCont.real)>0.01
        energyCont = np.around(energyCont[large_TB].real,decimals=4)
        TBparams = TBparams[large_TB]
        (orb1, orb2, trans1, trans2, trans3) = (orb1[large_TB], orb2[large_TB], trans1[large_TB], trans2[large_TB], trans3[large_TB])
        onsite_term = [(orb1==orb2) & (trans1==0) & (trans2==0) & (trans3==0)]
        notonsite_term = np.invert(onsite_term)[0]
        energyCont = energyCont[notonsite_term]
        TBparams = TBparams[notonsite_term]
        (orb1,orb2) = (orb1[notonsite_term],orb2[notonsite_term])
        (trans1,trans2,trans3) = (trans1[notonsite_term],trans2[notonsite_term],trans3[notonsite_term])
        sorting_ind = np.flip(np.argsort(np.abs(energyCont)))
        sortedEnergyCont = energyCont[sorting_ind]
        sort_TBparams = TBparams[sorting_ind]
        (Sorb1, Sorb2, Strans1, Strans2, Strans3) = (orb1[sorting_ind], orb2[sorting_ind], trans1[sorting_ind], trans2[sorting_ind], trans3[sorting_ind])
        stSorb1 = np.char.mod('%d', Sorb1)
        stSorb2 = np.char.mod('%d', Sorb2)
        #print("to string?", Sorb1, stSorb1)
        sorted_keys = np.char.add(stSorb1,stSorb2)

        sample_keys = sorted_keys# diff_evals_keys[sorted_sample_indices]
        sorted_sample_val = sortedEnergyCont #sample_val[sorted_sample_indices] # sorted_sample_val but with +/-
        sorted_sample_mag_val = np.abs(sortedEnergyCont)# np.flip(np.sort(abs(sample_val))) # This used to be sorted_sample_val
        #print(sorted_sample_val)
        #print(sample_keys[:20])
        #sample_params = diff_tb_params[sorted_sample_indices]
        #print(sample_params)
        (group_mag_vals, group_vals, bool_diff, count_list, indices) = ([sorted_sample_mag_val[0]], [sorted_sample_val[0]],
                                                                        [True], [], [])
        count = 1
        groups = []
        for i in range(1, np.size(sorted_sample_mag_val)):
            diff = abs(sorted_sample_val[i].real - sorted_sample_val[i - 1].real)
            if not diff < 0.005:
                group_vals.append(sorted_sample_val[i])
                group_mag_vals.append(sorted_sample_mag_val[i])
            bool_diff.append(diff >= 0.005)
        #print("bool_diff", bool_diff)
        for i in range(0, len(bool_diff) - 1):
            if not bool_diff[i + 1]:
                count += 1
            else:
                count_list.append(count)
                count = 1
        count_list.append(count)
        for n in range(len(group_vals)):
            groups.append((group_vals[n]*count_list[n], count_list[n]))
        for index in range(0, np.size(sorted_sample_mag_val)):
            if sorted_sample_mag_val[index] in group_mag_vals:
                indices.append(index)
        next_ind = []
        for i in range(len(bool_diff)):
            if bool_diff[i]:
                next_ind.append(i)
        next_ind = np.array(next_ind)
        first_vec_of_keys = []
        first_vec_of_orbs = []
        first_vec_of_trans = []
        first_vec_of_tbparams_uq = []
        first_vec_of_tbparams = []
        for i in range(len(next_ind)-1):
            first_vec_of_keys.append(np.unique(sample_keys[next_ind[i]:next_ind[i+1]]))
            first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[i]:next_ind[i + 1]].real,decimals=3)).tolist())
            first_vec_of_tbparams.append(sort_TBparams[next_ind[i]:next_ind[i + 1]].tolist())
            first_vec_of_orbs.append([Sorb1[next_ind[i]:next_ind[i + 1]].tolist(),Sorb2[next_ind[i]:next_ind[i + 1]].tolist()])
            first_vec_of_trans.append([Strans1[next_ind[i]:next_ind[i + 1]].tolist(),Strans2[next_ind[i]:next_ind[i + 1]].tolist(),Strans3[next_ind[i]:next_ind[i + 1]].tolist()])
        # don't forget to append the last element
        first_vec_of_keys.append(np.unique(sample_keys[next_ind[-1]:-1]))
        first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[-1]:-1].real,decimals=3)).tolist())
        first_vec_of_tbparams.append(sort_TBparams[next_ind[-1]:-1].tolist())
        first_vec_of_orbs.append([Sorb1[next_ind[-1]:-1].tolist(),Sorb2[next_ind[-1]:-1].tolist()])
        first_vec_of_trans.append([Strans1[next_ind[-1]:-1].tolist(),Strans2[next_ind[-1]:-1].tolist(),Strans3[next_ind[-1]:-1].tolist()])
        #print("First_vec_of_keys, ",first_vec_of_keys)
        groups = np.array(groups)
        one_d_vec_of_keys = []
        for arr in first_vec_of_keys:
            con = ''
            for ind, elem in enumerate(arr):
                if ind < 6:
                    con += elem + ", "
            one_d_vec_of_keys.append(con[0:-2])
        first_vec_keys_1D = np.array(one_d_vec_of_keys)
        first_vec_tbparams_1D = np.array(first_vec_of_tbparams_uq,dtype=object)
        #print(first_vec_tbparams_1D)

        #sort based on (energy change * num params) instead of just energy
        #print(np.abs(groups[:,0]))
        #print(first_vec_keys_1D)
        new_sort = np.flip(np.argsort(np.abs(groups[:,0])))

        #print("new sorting:",new_sort)

        self.groups = np.around(groups[new_sort],decimals=3)
        self.trans = np.array(first_vec_of_trans,dtype=object)[new_sort]
        self.tbparams = np.array(first_vec_of_tbparams,dtype=object)[new_sort]
        self.orbs = np.array(first_vec_of_orbs,dtype=object)[new_sort]
        self.coeffs = eigen_var #will use this to determine phase of orbitals
        #    print("orbs",self.orbs,"  trans",self.trans)
        self.keys = first_vec_keys_1D[new_sort]
        self.tbparams_uq= first_vec_tbparams_1D[new_sort]
        self.orig_oneTB = np.zeros(6)
        #print(first_vec_tbparams_1D[new_sort].T)
        if len(new_sort) < 6:
            loop_over = range(len(new_sort))
        else:
            loop_over = range(6)
        for param in loop_over:
            uq_params =  first_vec_tbparams_1D[new_sort][param]
            one_value = np.abs(uq_params[0])
            self.orig_oneTB[param] = one_value
        #self.orig_oneTB[:len(new_sort)] = first_vec_tbparams_1D[new_sort].T[0].T
        print("og TB", self.orig_oneTB)
        #self.new_evals = np.around(new_evals, decimals=5)
        return ([self.keys, self.groups, self.tbparams_uq])

    def plot_bond_run(self,ax=None,num_bond = 0):
        #print(self.orbs)
        orbs = self.orbs[num_bond]
        trans = self.trans[num_bond]
        tbparams = self.tbparams[num_bond]
        coeffs = self.coeffs
        coeffs[np.abs(coeffs)>0.01] = coeffs[np.abs(coeffs)>0.01]/np.abs(coeffs)[np.abs(coeffs)>0.01]
        phase = np.conj(coeffs[orbs[0][0]])*coeffs[orbs[1][0]]
        #print("phase from coeffs:",phase)
        kx = np.linspace(0,0.5,num=15)
        ky = np.linspace(0, 0.001, num=15)
        kz = np.linspace(0, 0.5, num=15)
        kpnts = np.array([kx,ky,kz]).T
        bond_run = np.zeros((15),dtype=np.complex_)
        for ind,k in enumerate(kpnts):
            for bond in range(len(tbparams)):
                vec = np.array([1,0,0])*trans[0][bond]+np.array([0,1,0])*trans[1][bond]+np.array([0,0,1])*trans[2][bond]
                vec = vec + self.orb_redcoords[orbs[1][bond]] - self.orb_redcoords[orbs[0][bond]]
                #print("check vec",vec)
                exp_fac = np.exp(-2j * np.pi * np.dot(k, vec))*np.conj(coeffs[orbs[0][bond]])*coeffs[orbs[1][bond]]
                bond_run[ind] = bond_run[ind] + tbparams[bond]*exp_fac
        bond_run = bond_run/2 # seems like it's twice as large as is should be
        bond_run = bond_run.real
        xaxis = np.linspace(0,1,num=15)
        # initialize figure and axis
        fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(111)
        else:
            ax = ax
        ax.plot(xaxis,bond_run)
        max = 2
        min = -2
        if (bond_run>2).any():
            max = np.amax(bond_run) + 1
        if (bond_run<-2).any():
            min = np.amin(bond_run) - 1
        ax.set_ylim(min, max)
        plt.close()
        return ax

    def change_sig_bonds(self,old_vals,new_vals,tbvals=None,num_bond=None):
        """To change tight-binding parameter, need to know two orbitals and three translations"""
        # if num_bond is set than tbvals need to be a float otherwise it is a list of floats

        #reassign all TB params with the same value as the one that has been changed
        chang_model = copy.deepcopy(self.newModel)
        chang_hops = copy.deepcopy(self._hoppings)

        for i in range(len(old_vals)):
            old_val = old_vals[i]
            new_val = new_vals[i]
            #print(old_val)
            same_hop = np.around(np.abs(chang_hops),decimals=3)==np.around(np.abs(old_val),decimals=3)
            #print(chang_hops[same_hop])
            chang_hops[same_hop] = new_val*np.sign(chang_hops[same_hop])
            #print(chang_hops[same_hop])
            chang_model.TB_params = chang_hops


        int_evals = []
        int_evecs = []
        for kpt in self.k_vec:
            (evals, evecs) = chang_model.get_ham(kpt)
            int_evals.append(evals)
            int_evecs.append(evecs)

        self.new_evals = np.array(int_evals).T

        '''
        # directly edit the original eigenvalues instead of recalculating
        if num_bond == None:
            if len(self.orbs) > 6:
                bonds = range(6)
            else:
                bonds = range(len(self.orbs))
            print("right length?",bonds)
        else:
            bonds = [num_bond]

        self.new_evals = copy.deepcopy(self.evals)
        for bond in bonds:
            orbs = self.orbs[bond]
            trans = self.trans[bond]
            if tbvals == None:
                # half the value
                tbparams = np.array(self.tbparams[bond]/2)
            else:
                #do 1-tbval because tbparams is substracted from evals
                tbparams = np.array(self.tbparams[bond])*(1-np.abs(np.array(tbvals[bond]/self.tbparams[bond][0])))
            for band in range(self.evals.shape[0]):
                for kind,kpt in enumerate(self.k_vec):
                    coeffs = self.evecs[band,kind]
                    for bnd in range(len(tbparams)):
                        orb1 = orbs[0][bnd]
                        orb2 = orbs[1][bnd]
                        vec = np.array([1, 0, 0]) * trans[0][bnd] + np.array([0, 1, 0]) * trans[1][bnd] + np.array([0, 0, 1]) * \
                              trans[2][bnd]
                        vec = vec + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                        # print("check vec",vec)
                        exp_fac = np.exp(-2j * np.pi * np.dot(kpt, vec)) * np.conj(coeffs[orb1]) * coeffs[orb2]
                        self.new_evals[band,kind] = self.new_evals[band,kind] - tbparams[bnd] * exp_fac
                    if abs(self.new_evals[band,kind].imag) > 0.00001:
                        print("error! have imaginary energy",band,kind,self.new_evals[band,kind])
        self.new_evals = np.sort(self.new_evals,axis=0)
        #self.new_evals = np.array(int_evals).T  # 2D array [n,k] or [n][k] where n is band index and k is kpt index
        '''

    def k_path(self,k_list, nk):
        # taken mostly from pythTB

        # must have more k-points in the path than number of nodes
        k_list = np.array(k_list)
        if nk < k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes = k_list.shape[0]

        # extract the lattice vectors from the TB model
        lat_per = np.copy([self._a1,self._a2,self._a3])
        # choose only those that correspond to periodic directions
        # lat_per = lat_per[self._per]
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node = np.zeros(n_nodes, dtype=float)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            k_node[n] = k_node[n - 1] + dklen

        # Find indices of nodes in interpolated list
        node_index = [0]
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)

        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist = np.zeros(nk, dtype=float)
        #   array listing the interpolated k-points
        k_vec = np.zeros((nk, 3), dtype=float)

        # go over all kpoints
        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            n_i = node_index[n - 1]
            n_f = node_index[n]
            kd_i = k_node[n - 1]
            kd_f = k_node[n]
            k_i = k_list[n - 1]
            k_f = k_list[n]
            for j in range(n_i, n_f + 1):
                frac = float(j - n_i) / float(n_f - n_i)
                k_dist[j] = kd_i + frac * (kd_f - kd_i)
                k_vec[j] = k_i + frac * (k_f - k_i)

        return (k_vec,k_dist,k_node)

class Widget(Bandstructure):#, CrystalOrbital):
    def __init__(self,wannierDir,wannierTag,numWannierOrbs,character,kpath = None, k_label = None):
        super(Widget,self).__init__(wannierDir,wannierTag,numWannierOrbs,kpath, k_label)
        self.character = character

    def plotWidget(self):
        plt.close()
        #self.plotBS()
        fig = plt.figure(figsize=(10, 8))
        self.text_ax = fig.add_axes([0.0,0.0,0.99,0.99])
        self.text_ax.axis('off')
        self.text_ax.text(0.665,0.815,"Important bonds",size=12)
        self.text_ax.text(0.38,0.93,"Wavefunction orbital character",size=12)
        self.ax1 = fig.add_axes([0.1,0.15,0.4,0.65])#add_subplot(131)
        self.ax_bond = fig.add_axes([0.55,0.15,0.4,0.45])
        self.ax_bond.set_title("Bond run")
        self.ax_bond.set_ylim(-5, 5)
        #intialize table of coefficients
        self.ax_table = fig.add_axes([0.2,0.915,0.7,0.2])
        self.ax_table.axis('off')
        self.ax2_table = fig.add_axes([0.52, 0.8, 0.36, 0.7])
        self.ax2_table.axis('off')

        #make button to recalculate bandstructure


        #make textboxes stuff
        butt_ax = fig.add_axes([0.88, 0.631, 0.1, 0.021])
        butt = Button(butt_ax,"calculate!")

        def recalc_bs(val):
            print("been clicked!",val)
            hold = [text_box1.text,text_box2.text,text_box3.text,text_box4.text,text_box5.text,text_box6.text]
            new_vals = np.array([float(i) for i in hold])
            print(new_vals)
            old_vals = self.orig_oneTB
            self.change_sig_bonds(old_vals=old_vals, new_vals=new_vals)
            self.ax1.clear()
            self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
            plt.draw()

        butt.on_clicked(recalc_bs)
        butt_ax._button = butt
        axbox = fig.add_axes([0.88, 0.78, 0.1, 0.023])
        text_box = TextBox(axbox, "","new TB params")
        axbox1 = fig.add_axes([0.88, 0.757, 0.1, 0.021])
        text_box1 = TextBox(axbox1, "","")
        axbox2 = fig.add_axes([0.88, 0.736, 0.1, 0.021])
        text_box2 = TextBox(axbox2, "","")
        axbox3 = fig.add_axes([0.88, 0.715, 0.1, 0.021])
        text_box3 = TextBox(axbox3, "","")
        axbox4 = fig.add_axes([0.88, 0.694, 0.1, 0.021])
        text_box4 = TextBox(axbox4, "","")
        axbox5 = fig.add_axes([0.88, 0.673, 0.1, 0.021])
        text_box5 = TextBox(axbox5, "","")
        axbox6 = fig.add_axes([0.88, 0.652, 0.1, 0.021])
        text_box6 = TextBox(axbox6, "","")

        '''
        self.test_TB = [0,0,0,0,0,0] #self.orig_oneTB
        def submit1(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[0]
            new_val = tbvalue
            self.test_TB[0] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=0)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box1.on_submit(submit1)
        def submit2(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[1]
            new_val = tbvalue
            self.test_TB[1] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=1)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box2.on_submit(submit2)
        def submit3(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[2]
            new_val = tbvalue
            self.test_TB[2] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=2)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box3.on_submit(submit3)
        def submit4(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[3]
            new_val = tbvalue
            self.test_TB[3] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=3)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box4.on_submit(submit4)
        def submit5(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[4]
            new_val = tbvalue
            self.test_TB[4] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=4)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box5.on_submit(submit5)
        def submit6(tbvalue):
            tbvalue = float(tbvalue)
            old_val = self.orig_oneTB[5]
            new_val = tbvalue
            self.test_TB[5] = tbvalue
            if old_val != 0:
                self.change_sig_bonds(old_val = old_val,new_val=new_val)#tbvals = self.test_TB)
                self.ax_bond.clear()
                self.ax_bond.set_title("Bond run")
                self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=5)
                self.ax1.clear()
                self.ax1 = self.plotBS(ax=self.ax1,selectedDot=self.selectedDot,plotnew=True)
                plt.draw()
        text_box6.on_submit(submit6)
        '''
        #make info for tables
        #self.ax_table.axis('tight')
        fig.suptitle('Visualize Bandstructure Chemistry',size = 15)
        self.ax1 = self.plotBS(ax=self.ax1)
        rows = ['Wannier Orbital', 'Orbital Type', '% coefficient']
        self.table_info = np.array([['     ', '', '','', '','', '',''], ['', '', '','', '','', '',''], ['', '', '','', '','', '','']])
        self.table2_info = np.array([['        ','','',''],['','','',''],['','','',''],['','','',''],['','','',''],['','','','']])
        cols = ['Orbitals', 'Tot Energy', '# Params', 'TB params']
        self.table = self.ax_table.table(cellText=self.table_info, rowLabels=rows, cellLoc='center')
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(10)
        self.table_keygroup = self.ax2_table.table(cellText=self.table2_info, colLabels=cols, cellLoc='center')
        self.table_keygroup.auto_set_font_size(False)
        self.table_keygroup.set_fontsize(10)
        self.ax_table.set_title('Crystal Orbital Composition')
        evals = self.evals
        evecs = self.evecs
        kvec = self.k_vec
        num = self.numOrbs
        character = self.character

        def onpick3(event):
            thisline = event.artist
            kpoint = event.ind[0]
            band = int(thisline.get_label())
            wanOrbs = []
            combo = []
            count = 0
            #reset table
            for row in range(0, 3):
                for column in range(0, 8):
                    self.table_info[row][column] = ''
            # reset second table
            for row2 in range(0,6):
                for column2 in range(0,4):
                    self.table2_info[row2][column2] = ''

            #get percent of each orbital in the total wavefunction and put in table
            count = 0
            for i in range(0,num): #i loops over orbitals
                #old evecs arrangement: evecs[band][kpoint][0][i]
                if abs(evecs[band][kpoint][i]) > 0.1: #only include orbitals present at point
                    #populate table
                    self.table_info[0,count] = i+1 # wannier orbital number
                    self.table_info[1, count] = character[i] # cooresponding orbital character
                    evec_val = evecs[band][kpoint][i]
                    #print(evec_val)
                    self.table_info[2, count] = (evec_val.real**2 + evec_val.imag**2)*100 # percent of orbital at point
                    count = count+1
                    #save for input in getCrystalOrbital
                    wanOrbs.append(i+1)
                    combo.append(evec_val)

            print("kpoint:",kvec[kpoint],"kpoint num:",kpoint, "band:", band, "energy:",evals[band][kpoint])
            #self.getCrystalOrbital(kvec[kpoint],desiredOrbs=wanOrbs,orbitalCombo=combo)
            #replot figure
            self.ax1.clear()

            #update table
            for row in range(0, 3):
                for column in range(0, 8):
                    self.table.get_celld()[row, column].get_text().set_text(self.table_info[row][column])
            #plt.draw()
            # update the second table
            keys_and_groups = self.get_significant_bonds(band=band, kpoint=kpoint)
            #self.change_sig_bonds(tbvals=0.3,num_bond=0)
            self.ax_bond.clear()
            self.ax_bond.set_title("Bond run")
            self.ax_bond = self.plot_bond_run(ax=self.ax_bond)
            self.selectedDot = [kpoint,band]
            self.ax1 = self.plotBS(ax=self.ax1,selectedDot=[kpoint,band])
            for row2 in range(0, 6):
                if row2 >= len(keys_and_groups[0]):
                    self.table_keygroup.get_celld()[row2 + 1, 0].get_text().set_text('')
                    self.table_keygroup.get_celld()[row2 + 1, 1].get_text().set_text('')
                    self.table_keygroup.get_celld()[row2 + 1, 2].get_text().set_text('')
                    self.table_keygroup.get_celld()[row2 + 1, 3].get_text().set_text('')
                    #self.test_TB[row2] = 0
                else:
                    self.table_keygroup.get_celld()[row2 + 1, 0].get_text().set_text(str(keys_and_groups[0][row2]))
                    self.table_keygroup.get_celld()[row2 + 1, 1].get_text().set_text(str(keys_and_groups[1][row2][0]))
                    self.table_keygroup.get_celld()[row2 + 1, 2].get_text().set_text(str(keys_and_groups[1][row2][1]))
                    self.table_keygroup.get_celld()[row2 + 1, 3].get_text().set_text(str(keys_and_groups[2][row2]))
                    #self.test_TB[row2] = keys_and_groups[2][row2][0]
            text_box1.text_disp.set_text(str(abs(self.orig_oneTB[0])))
            text_box2.text_disp.set_text(str(abs(self.orig_oneTB[1])))
            text_box3.text_disp.set_text(str(abs(self.orig_oneTB[2])))
            text_box4.text_disp.set_text(str(abs(self.orig_oneTB[3])))
            text_box5.text_disp.set_text(str(abs(self.orig_oneTB[4])))
            text_box6.text_disp.set_text(str(abs(self.orig_oneTB[5])))
            plt.draw()

        fig.canvas.mpl_connect('pick_event', onpick3)

        #text_box1._rendercursor()
        #plt.show()
        return fig

def _cart_to_red(tmp,cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    #  ex: prim_coord = _cart_to_red((a1,a2,a3),cart_coord)
    (a1,a2,a3)=tmp
    # matrix with lattice vectors
    cnv=np.array([a1,a2,a3])
    # transpose a matrix
    cnv=cnv.T
    # invert a matrix
    cnv=np.linalg.inv(cnv)
    # reduced coordinates
    red=np.zeros_like(cart,dtype=float)
    for i in range(0,len(cart)):
        red[i]=np.dot(cnv,cart[i])
    return red

def _red_to_cart(prim_vec,prim_coord):
    """
    :param prim_vec: three float tuples representing the primitive vectors
    :param prim_coord: list of float tuples for primitive coordinates
    :return: list of float tuples for cartesian coordinates
            ex: cart_coord = _red_to_cart((a1,a2,a3),prim_coord)
    """
    (a1,a2,a3)=prim_vec
    prim = prim_coord
    # cartesian coordinates
    cart=np.zeros_like(prim_coord,dtype=float)
    for i in range(0,len(cart)):
        cart[i,:]=a1*prim[i][0]+a2*prim[i][1]+a3*prim[i][2]
    return cart

def normalize_wf(wavefunc,min,max,gridnum):

    integral = periodic_integral_3d(np.conj(wavefunc)*wavefunc,min,max,gridnum)
    print("integral:",(integral)**(1/2))
    return wavefunc/(integral)**(1/2)

def periodic_integral_3d(f,a,b,n):
    func = f #3D float array
    mini = a #[0,0,0]
    maxi = b #[ax,ay,az]
    grid = n #[24,24,33]
    #max_val = np.max(func)
    #print("max val:",max_val)
    afterz_integral = np.zeros((grid[0],grid[1]))
    aftery_integral = np.zeros((grid[0],1))
    for x in range(grid[0]):
        for y in range(grid[1]):
            afterz_integral[x,y] = periodic_integral(func[x,y,:],mini[2],maxi[2],grid[2])
    for x in range(grid[0]):
        aftery_integral[x] = periodic_integral(afterz_integral[x],mini[1],maxi[1],grid[1])
    total_integral = periodic_integral(aftery_integral,mini[0],maxi[0],grid[0])
    #substract = max_val*(maxi[0]-mini[0])/grid[0]*(maxi[1]-mini[1])/grid[1]*(maxi[2]-mini[2])/grid[2]
    #print("substracted:", substract)
    return total_integral  #- substract

def periodic_integral(f, a, b, n):
    h = (b-a)/n
    integral = (f[0]+f[n-1])*0.5
    for i in range(n-1):
        integral += (f[i]+f[i+1])*0.5
    return integral*h

#character = ['Pb s','Pb d','Pb d','Pb d','Pb d','Pb d','Pb s','Pb d','Pb d','Pb d','Pb d','Pb d','O s','O pz', 'O py', 'O px','O s','O pz', 'O py', 'O px'] # for valence bands of PbO
#character = ['Pb pz','Pb py', 'Pb px','Pb pz','Pb py', 'Pb px'] # for conduciton bands of PbO
