from cellpose import models, train
from pygor.load import Core




# Prepare your training data
# images = 

# Train custom model
model = models.CellposeModel(gpu=True, model_type='cyto')  # or start from scratch
new_model_path = train.train_seg(
    model.net,
    train_data=images,
    train_labels=masks,
    test_data=test_images,
    test_labels=test_masks,
    save_path='./models/',
    n_epochs=100
)