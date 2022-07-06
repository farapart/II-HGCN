This is the code of paper "Intra and Inter Domain HyperGraph Convolutional Network for Cross-Domain Recommendation".

To run this code, you need to do the following process to generate the preprocessed matrices. So that the training process of the model will be faster.

### Dataset

To save time in uploading and downloading, we upload the dataset of Movie & Book, Movie & Music, and Music & Book to:

https://pan.baidu.com/s/13vveAcz0KrR1k14Sr2X4iA  

Extraction Code: 86vt

After downloading these datasets. You just need to put the "processed_data" folder under the "II-HGCN" folder.


### Steps
1. Initialize the variable "dataset_string" in utils.py. (amazon: Movie & Book. amazon2: Movie & Music. amazon3: Music & Book)
2. Run the main function in utils.py.
3. Then you can run the code as follows.

### Run the code

1. Run code in Movie & Book dataset: nohup python -u main.py --dataset amazon
2. Run code in Movie & Music dataset: nohup python -u main.py --dataset amazon2
3. Run code in Music & Book dataset: nohup python -u main.py --dataset amazon3
