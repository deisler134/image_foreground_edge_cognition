img_rows, img_cols = 320, 320
# img_rows_half, img_cols_half = 160, 160
channel = 4
batch_size = 16
epochs = 1000
patience = 50
num_samples = 30000     #1393    #43100
num_train_samples = 12000       #1293     #34480
# num_samples - num_train_samples
num_valid_samples = 1500        #100      #8620
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

##############################################################
# Set your paths here

# path to provided foreground images
fg_path = '/media/deisler/Data/project/coco/cocodata/Matting_data/fg/'   #'data/fg/'

# path to provided alpha mattes
a_path = '/media/deisler/Data/project/coco/cocodata/Matting_data/mask/'  ##'data/fg/'

# Path to background images (MSCOCO)
bg_path = '/media/deisler/Data/project/coco/cocodata/Matting_data/fg/'    #data/bg/'

# Path to folder where you want the composited images to go
out_path = 'data/merged/'

##############################################################
