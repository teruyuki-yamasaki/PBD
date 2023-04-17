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

- dataframe 
```
train = pd.read_csv(INPUT/"train.csv")
test = pd.read_csv(INPUT/"test.csv")

print(len(set(train["image"].unique()) & set(test["image"].unique()))) # 重複数 0
print(train["image"].nunique()) # 訓練画像数 
print(test["image"].nunique()) # テスト画像数
print(train["Instance ID"].max()) # Instance ID (各画像におけるannotationのID) の最大値
```
