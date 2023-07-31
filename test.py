# Defining a custom dataset class for loading and preprocessing image data
class manualDataset(Dataset):
    def __init__(self, file_path, split='train', val_split=False):
        df = pd.read_csv(file_path)
        self.split = split
        self.val_split = val_split
        if self.split=='test' and val_split==False :
            # test data
            self.raw_files = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.labels = None
        else:
            # training data
            self.raw_files = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.labels = torch.from_numpy(df.iloc[:,0].values)
        
        if(val_split):
            self.raw_files = self.raw_files[:10]
          
       
        # Defining the transformations to be applied to the images
        self.train_transform = transforms.Compose([ transforms.ToPILImage(),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.test_transform = transforms.Compose([ transforms.ToTensor()])
          

        
    def __len__(self):
        return len(self.raw_files)
    
    def __getitem__(self, idx):
        # Reading in the image file
        raw = self.raw_files[idx]
      
        
        # Applying the appropriate transformation based on the split
        if self.split == 'train' or self.val_split:
            raw = self.train_transform(raw)
            label = self.labels[idx]
            return raw, label
        elif self.split == 'test':
            raw = self.test_transform(raw)
            return raw