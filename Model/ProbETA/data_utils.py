
import numpy as np
import torch, h5py, os
from collections import namedtuple
from collections import Counter
import random
import torch.distributions as dist
import math

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=0):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

#xs = np.array([np.r_[1], np.r_[1, 2, 3], np.r_[2, 1]])


def mark_original_data(data):
    first_positions = []
    unique_elements = torch.unique(data)
    for element in unique_elements:
        first_positions.append((data == element).nonzero(as_tuple=False).min().item())
    return first_positions

def calculate_similarity_matrices(devid):

    same_dev_mask = (devid.unsqueeze(0)==devid.unsqueeze(1)).int()
    return same_dev_mask

def multivariate_gaussian_nll_loss(predicted_mean, predicted_covariance, observed_values):
    predicted_covariance=(predicted_covariance+predicted_covariance.T)/2
    mvn = dist.MultivariateNormal(predicted_mean, predicted_covariance)
    likelihood=mvn.log_prob(observed_values)
    return -likelihood

def load_data(timeDis,datapath,timeslot,timespan):
    files = list(filter(lambda x: x.endswith(".h5"), sorted(os.listdir(datapath))))
    if timeDis:
        dataloader = DataLoader(datapath,1,timeslot,timespan)
    else:
        dataloader = DataLoader(datapath,1)
        
    print("Loading the data...")
    dataloader.read_files(files)
    train_slot_size = np.array(list(map(lambda s: s.ntrips, dataloader.slotdata_pool_train)))
    test_slot_size = np.array(list(map(lambda s: s.ntrips, dataloader.slotdata_pool_test)))
    print("There are {} training trips in total".format(train_slot_size.sum()))  
    print("There are {} testing trips in total".format(test_slot_size.sum()))  
        
    
    return dataloader,train_slot_size,test_slot_size




def adjust_lr(optimizer, epoch, lr, lr_decay):
    lr = max(lr * lr_decay,0.000001)
    print("lr:",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def pearson_correlation(x, y):

    predictions=torch.tensor(x)
    targets=torch.tensor(y)
    
    residuals = targets - predictions
    residuals_squared = torch.pow(residuals, 2)
    residuals_sum = torch.sum(residuals_squared)

    mean_targets = torch.mean(targets)
    total_sum_of_squares = torch.sum(torch.pow(targets - mean_targets, 2))
    r_squared = 1 - (residuals_sum / total_sum_of_squares)
    
    return r_squared


class CRPSMetric:
    def __init__(self, x, loc, scale):
        self.value = x
        self.loc = loc
        self.scale = torch.sqrt(scale)
    def gaussian_pdf(self, x):
        """
        Probability density function of a univariate standard Gaussian distribution with zero mean and unit variance.
        """
        _normconst = 1.0 / math.sqrt(2.0 * math.pi)
        return _normconst * torch.exp(-(x * x) / 2.0)
    def gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    def gaussian_crps(self):
        sx = (self.value - self.loc) / self.scale
        pdf = self.gaussian_pdf(sx)
        cdf = self.gaussian_cdf(sx)
        pi_inv = 1.0 / math.sqrt(math.pi)
        # the actual crps
        crps = self.scale * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SlotData():
    def __init__(self, trips, times, ratios, S, distances,devid=None, maxlen=100):
        """
        trips (n, *): each element is a sequence of road segments
        times (n, ): each element is a travel cost
        ratios (n, 2): end road segments ratio
        S (138, 148) or (num_channel, 138, 148): traffic state matrix
        """
        ## filter out the trips that are too long
        self.devid=devid
        idx = [i for (i, trip) in enumerate(trips) if len(trip) <= maxlen]

        '''
        self.trips = trips[idx]
        self.times = times[idx]
        if devid is not None:
            self.devid = devid[idx]
        self.ratios = ratios[idx]
        self.distances = distances[idx]
        '''
        self.S = torch.tensor(S, dtype=torch.float32)
        ## (1, num_channel, height, width)
        if self.S.dim() == 2:
            self.S.unsqueeze_(0).unsqueeze_(0)
        elif self.S.dim() == 3:
            self.S.unsqueeze_(0)
        ## re-arrange the trips by the length in reverse order
        
        #idx = argsort(trips)
        self.trips = trips[idx]
        self.times = torch.tensor(times[idx], dtype=torch.float32)
        if devid is not None:
            self.devid = torch.tensor(devid[idx], dtype=torch.float32)
        self.ratios = torch.tensor(ratios[idx], dtype=torch.float32)
        self.distances = torch.tensor(distances[idx], dtype=torch.float32)

        self.ntrips = len(self.trips)
        self.start = 0


    def random_emit(self, batch_size):
        """
        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        SD = namedtuple('SD', ['trips', 'times','devid', 'ratios', 'S', 'distances'])
        start = np.random.choice(max(1, self.ntrips-batch_size+1))
        end = min(start+batch_size, self.ntrips)

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        if self.devid is not None:
            devid = self.devid[start:end]
        else:
            devid=0
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times,devid=devid, ratios=ratios, S=self.S, distances=distances)

    def order_emit(self, batch_size,stepinput=None):
        """
        Reset the `start` every time the current slot has been traversed
        and return none.

        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        if stepinput:
            step=stepinput
        else:
            step=batch_size
        if self.start >= self.ntrips:
            self.start = 0
            return None
        SD = namedtuple('SD', ['trips', 'times','devid', 'ratios', 'S', 'distances'])
        start = self.start
        end = min(start+batch_size, self.ntrips)
        self.start += step

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        if self.devid is not None:
            devid = self.devid[start:end]
        else:
            devid=0
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times,devid=devid, ratios=ratios, S=self.S, distances=distances)


class DataLoader():
    def __init__(self, trainpath,train=0,timeslot=None,timespan=None, num_slot=71):
        """
        trainpath (string): The h5file path
        num_slot (int): The number of slots in a day
        """
        self.trainpath = trainpath
        self.num_slot = num_slot
        self.timeslot=timeslot
        self.timespan=timespan
        self.slotdata_pool_train = []
        self.slotdata_pool_test = []
        ## `weights[i]` is proportional to the number of trips in `slotdata_pool[i]`
        self.weights = None
        ## The length of `slotdata_pool`
        self.length = 0
        ## The current index of the order emit
        self.order_idx_train = -1
        self.order_idx_test = -1
        self.train=train
        self.linkDic=Counter()

        
    def read_file(self, fname,trainLink=None):
        """
        Reading one h5file and appending the data into `slotdata_pool`. This function
        should only be called by `read_files()`.
        """
        
        pass_counter=0
        
        if self.timeslot!=None and self.timespan!=None:
            timeslotstart=self.timeslot
            timeslotend=self.timeslot+self.timespan
        else:
            timeslotstart=1
            timeslotend=self.num_slot+1
        
        with h5py.File(fname,'r') as f:
            for slot in range(timeslotstart, timeslotend):
                S = np.rot90(f["/{}/S".format(slot)][...]).copy()
                n = f["/{}/ntrips".format(slot)][...]
                if n == 0: continue
                trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]                                                     
                times = [f["/{}/time/{}".format(slot, i)][...] for i in range(1, n+1)]
                if self.train:
                    devid = [int(f["/{}/devid/{}".format(slot, i)][...]) for i in range(1, n+1)]
                    
                ratios = [f["/{}/ratio/{}".format(slot, i)][...] for i in range(1, n+1)]
                distances = [f["/{}/distance/{}".format(slot, i)][...] for i in range(1, n+1)]
                
                
                ############################分集################################
                tripNum=len(trips)

                # 生成100以内的所有数字
                all_indices = list(range(tripNum))

                # 随机选取85个作为索引1
                index1 = random.sample(all_indices, int(tripNum*0.85))

                # 选取剩余的15个作为索引2
                index2 = list(set(all_indices) - set(index1))
                
                
                
                train_trips=np.array(trips)[index1]
                train_times=np.array(times)[index1]
                train_ratios=np.array(ratios)[index1]
                train_distances=np.array(distances)[index1]
                train_devid=np.array(devid)[index1]
                
                test_trips=np.array(trips)[index2]
                test_times=np.array(times)[index2]
                test_ratios=np.array(ratios)[index2]
                test_distances=np.array(distances)[index2]
                test_devid=np.array(devid)[index2]
                
                
                
                
                if self.train:
                    self.slotdata_pool_train.append(SlotData(np.array(train_trips), np.array(train_times), np.array(train_ratios),S,np.array(train_distances),np.array(train_devid))) 
                    self.slotdata_pool_test.append(SlotData(np.array(test_trips), np.array(test_times), np.array(test_ratios),S,np.array(test_distances),np.array(test_devid))) 
                else:
                    self.slotdata_pool_train.append(SlotData(np.array(train_trips), np.array(train_times), np.array(train_ratios), S,np.array(train_distances)))
                    self.slotdata_pool_test.append(SlotData(np.array(test_trips), np.array(test_times), np.array(test_ratios), S,np.array(test_distances)))

        
        
        
        
        
    def read_files(self, fname_lst,trainLink=None):
        """
        Reading a list of h5file and appending the data into `slotdata_pool`.
        """
        for fname in fname_lst:
            fname = os.path.basename(fname)
            print("Reading {}...".format(fname))
            self.read_file(os.path.join(self.trainpath, fname),trainLink)
            print("Done.")
        self.weights_train = np.array(list(map(lambda s:s.ntrips, self.slotdata_pool_train)))
        self.weights_train = self.weights_train / np.sum(self.weights_train)
        self.length_train = len(self.weights_train)
        
        self.weights_test = np.array(list(map(lambda s:s.ntrips, self.slotdata_pool_test)))
        self.weights_test = self.weights_test / np.sum(self.weights_test)
        self.length_test = len(self.weights_test)
        #self.order = np.random.permutation(self.length)
        self.order_train = np.arange(self.length_train)
        self.order_idx_train = 0
        self.order_test = np.arange(self.length_test)
        self.order_idx_test = 0

    def random_emit(self, batch_size):
        """
        Return a batch of data randomly.
        """
        i = np.random.choice(self.length, p=self.weights)
        
        return self.slotdata_pool[i].random_emit(batch_size),i

    def order_emit_train(self, batch_size,stepinput=None):
        """
        Visiting the `slotdata_pool` according to `order` and returning the data in the
        slot `slotdata_pool[i]` orderly.
        """
        i = self.order_train[self.order_idx_train]
        data = self.slotdata_pool_train[i].order_emit(batch_size,stepinput)
        if data is None: ## move to the next slot
            self.order_idx_train += 1
            if self.order_idx_train >= self.length_train:
                self.order_idx_train = 0
                #self.order = np.random.permutation(self.length)
            i = self.order_train[self.order_idx_train]
            data = self.slotdata_pool_train[i].order_emit(batch_size,stepinput)
        return data,i
    
    
    
    def order_emit_test(self, batch_size,stepinput=None):
        """
        Visiting the `slotdata_pool` according to `order` and returning the data in the
        slot `slotdata_pool[i]` orderly.
        """
        i = self.order_test[self.order_idx_test]
        data = self.slotdata_pool_test[i].order_emit(batch_size,stepinput)
        if data is None: ## move to the next slot
            self.order_idx_test += 1
            if self.order_idx_test >= self.length_test:
                self.order_idx_test = 0
                #self.order = np.random.permutation(self.length)
            i = self.order_test[self.order_idx_test]
            data = self.slotdata_pool_test[i].order_emit(batch_size,stepinput)
        return data,i
