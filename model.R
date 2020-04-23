# CNN
setwd("/home/sergio/Projects/Agro/postcovid19/")


library(EBImage)
library(tidyverse)
library(keras)
library(tfdatasets)

data_dir_train <- "/home/sergio/Projects/Agro/postcovid19/data/train/"
data_dir_valid <- "/home/sergio/Projects/Agro/postcovid19/data/valid/"
data_dir_test <- "/home/sergio/Projects/Agro/postcovid19/data/test/"

# generators
train_data_gen = image_data_generator(
  rescale = 1/255
)

valid_data_gen = image_data_generator(
  rescale = 1/255
)


test_data_gen <-image_data_generator(
  rescale = 1/255
)

# define variables
classes <- list.dirs(data_dir_train,full.names = FALSE,recursive = TRUE)%>%
  as.data.frame()

classes <- classes[-c(1),]%>%
  as.list()

indices <- as.data.frame(classes)

output_n <- length(classes)

img_heigt <- 20
img_weiht <- 20
batch_size <- 32
epochs <- 15
target_size <- c(img_heigt,img_weiht)
chanels <- 3


# training images generator
train_image_array_gen <- flow_images_from_directory(data_dir_train, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = classes,
                                                    seed = 42)

plant_classes_indices <- train_image_array_gen$class_indices %>%
  as.data.frame()

df.classes <- t(plant_classes_indices) %>%
  as.data.frame()

df.classes$classes <- rownames(df.classes)
rownames(df.classes) = df.classes$V1

write_csv(df.classes,"/home/sergio/Projects/Agro/postcovid19/classes.csv")

# validation images generator 
valid_image_array_gen <- flow_images_from_directory(data_dir_valid, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = classes,
                                                    seed = 42)



test_image_array_gen <- flow_images_from_directory(directory = data_dir_test,
                                                   generator = test_data_gen,
                                                   target_size = target_size,
                                                   seed = 42
                                                   
                                                   
)

#define number of training samples
train_samples <- train_image_array_gen$n
valid_samples <- valid_image_array_gen$n



#defining model
model <- keras_model_sequential() %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_weiht, img_heigt, chanels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>% 
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")


# compile
model%>%compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# fit and training 
model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
)










model <- load_model_tf("/home/sergio/Projects/Agro/postcovid19/model/cnn-agro/")


#predict
#preproces image 

test_image = image_load("/home/sergio/Projects/Agro/postcovid19/data/train/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG",
                        target_size = target_size)  %>%
                                    image_to_array() %>%
                                            array_reshape(dim = c(1,20,20,3))
test_image = test_image * 1/255 #important

preds <- model %>% predict_proba(test_image) %>%
  as.data.frame() %>%
  t() %>%
  as.data.frame() %>%
  format(.,scientific = F)
classes <- read.csv("./classes.csv")
rownames(preds) <- classes$classes
names(classes) <- "score"
View(preds)

#predict


model %>% predict_proba(test_image)

#save model

save_model_tf(model, "cnn-agro")


