from keras import models

# Load model
siamese_model = models.load_model('siamese_model.keras')

siamese_model.summary()

