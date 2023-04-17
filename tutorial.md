# Tutorial 

## flow 
- config 
```
TRAIN_PATH = Path("/content/train_imgs")
TEST_PATH = Path("/content/test_imgs")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
USE_SCHEDULER = False
NUM_EPOCHS = 3
...
```

- seed
```
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(SEED)
```

- check dataframe 
```
train = pd.read_csv(INPUT/"train.csv")
test = pd.read_csv(INPUT/"test.csv")

print(len(set(train["image"].unique()) & set(test["image"].unique()))) # 重複数 0
print(train["image"].nunique()) # 訓練画像数 
print(test["image"].nunique()) # テスト画像数
print(train["Instance ID"].max()) # Instance ID (各画像におけるannotationのID) の最大値
```

- edit dataframe 
```
annotation_df = train.groupby('image')['Instance ID'].max().reset_index()
remove_imgname = annotation_df[annotation_df['Instance ID'] >= 500]['image'].tolist()
train = train[~train['image'].isin(remove_imgname)].reset_index(drop = True)
```

- dataset 
```
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target
        
class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

class Compose:
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    if NORMALIZE:
        transforms.append(Normalize())

    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)
```
