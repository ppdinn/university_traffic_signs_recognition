import pandas as pd
from PIL import Image
import os

df = pd.read_csv('train.csv')
print('Total samples:', len(df))
print('Number of classes:', df['class_number'].nunique())
classes = sorted(df['class_number'].unique())
print('Classes:', classes)
vc = df['class_number'].value_counts().sort_index()
print('Min samples per class:', vc.min())
print('Max samples per class:', vc.max())
print('Mean samples per class:', vc.mean())
print('Median samples per class:', vc.median())

train_dir = 'train/train'
train_files = os.listdir(train_dir)
print(f'\nTotal train images: {len(train_files)}')

img = Image.open(os.path.join(train_dir, train_files[0]))
print(f'Image size: {img.size}, mode: {img.mode}')

test_dir = 'test/test'
test_files = os.listdir(test_dir)
print(f'Total test images: {len(test_files)}')
