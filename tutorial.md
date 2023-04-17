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


```
class SemiDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None, resize=False):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df

        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH
        
        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby('image')['rle'].agg(lambda x: list(x)).reset_index()
        temp_df = temp_df.merge(train[["image","height", "width"]].drop_duplicates(), on="image", how="inner").reset_index(drop=True)
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                    'image_id': row['image'],
                    'path': self.image_dir / row['image'],
                    'annotations': row["rle"],
                    'height': row["height"],
                    'width': row["width"]
                    }
    
    def get_box(self, a_mask):
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["path"]
        img = Image.open(img_path).convert("RGB")
        
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        info = self.image_info[idx]
        
        n_objects = len(info['annotations'])
   
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []
        
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (info['height'], info['width']))
            a_mask = Image.fromarray(a_mask)
            
            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            
            boxes.append(self.get_box(a_mask))

        labels = [1 for _ in range(n_objects)]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
```
