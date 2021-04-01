import os

data_path = os.path.join('..', 'data')
vocab_path = os.path.join(data_path, 'vocab')
annotation_path = os.path.join(data_path, 'preprocessed')
feat_path = os.path.join(data_path, 'feat')

video_feat_path = os.path.join(data_path, 'video_feat')

java_path = '/usr/local/lib/jdk1.8.0_241/bin/java'
os.environ['PATH'] += ':' + os.path.split(java_path)[0]
