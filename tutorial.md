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
