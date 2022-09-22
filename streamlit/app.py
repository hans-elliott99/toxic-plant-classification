import streamlit as st

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from PIL import Image
import pillow_heif  ##for processing heif/heic format images (sometimes produced by phones)

from filesplit.merge import Merge ##for mergine split models

# ----Streamlit Settings and Style ----
st.set_page_config(page_title="Herbarium Classification", page_icon=None,
                  layout="centered", 
                  initial_sidebar_state="auto", 
                  menu_items=None)


# Helpers ---
@st.cache(allow_output_mutation=True)
def load_model():
    """Recombines split models and loads them into memory."""
    # Compile split models
    merge1 = Merge('./streamlit/split_model_1/', './streamlit/split_model_1/', 'compiled_model1.h5',)
    merge1.merge(cleanup=False) ##keep split files with False
    merge2 = Merge('./streamlit/split_model_2/', './streamlit/split_model_2/', 'compiled_model2.h5')
    merge2.merge(cleanup=False) 

    # Load models
    bestmod = './streamlit/split_model_1/compiled_model1.h5'
    finalmod = './streamlit/split_model_2/compiled_model2.h5'
    bestmod = tf.keras.models.load_model(bestmod, compile=False)
    finalmod = tf.keras.models.load_model(finalmod, compile=False)    
    return bestmod, finalmod

def upload_predict(image, models, image_size=(199,199)):
    """Accepts the uploaded image (from PIL), passes through the model and organizes the results for display."""
    image = np.array(image)
    if len(image.shape) > 2 and image.shape[2] == 4:
        #convert the image from RGBA2RGB (for example, if input is PNG)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = prep_input_image(image, resize=image_size)

    pred1 = models[0](image, training=False)
    pred2 = models[1](image, training=False)
    prediction = np.mean([pred1, pred2]) ##ensemble
    result_df, result = preds_to_results(pred=prediction)
    # pred_prob = [p for p in top5['prob']]
    # category, scientific_name, family, genus, species = query_plant_info(pred_classes)
    # out = pd.DataFrame({
    #     'category_id' : category,
    #     'confidence' : pred_prob,
    #     'scientific name' : scientific_name,
    #     'family' : family,
    #     'genus' : genus,
    #     'species' : species
    # })
    if prediction.item() < 0.5:
        confidence = (1-prediction.item()) * 100
    else:
        confidence = prediction.item() * 100
    out_table = pd.DataFrame({
        'Prediction' : result,
        'Confidence' : [f'{round(confidence, 3)} %'],
        # 'ensemble': [[pred1.numpy().item(), pred2.numpy().item()]]
    })
    if result[0]=="Toxic": 
        out_text = "This could be an image of poison oak, poison ivy, or poison sumac."
    else:
        out_text = "The model predicts that this is not a rash plant, but proceed with caution."

    return out_table, out_text


def prep_input_image(image, resize=(199, 199)):
    """Applies the preprocessing function and standardizes the image prior to inference."""
    image = image.astype('float32')
    # Preprocessing
    image = np.array(image)
    image = preprocess_image(image, image_size=resize, resize=True, recolor=False) ##PIL reads in as RGB automatically
    # Rescale pixels
    image *= 1.0/255
    # standardize image - https://github.com/keras-team/keras/issues/2559
    image -= np.mean(image, axis=2, keepdims=True) ##samplewise center
    image /= (np.std(image, axis=2, keepdims=True) + 1e-7) ##samplewise std normalization
    # Reshape dimensions (add batch dimension)
    return np.expand_dims(image, axis=0)

def preprocess_image(img, bright_threshold=0.25, bright_value=30, image_size=(199,199), resize=True, recolor=True):
    """
    Applies a standardized preprocessing procedure to every image before it is passed into the model.
    Notes:
    If using tf.keras.preprocessing.image.ImageDataGenerator (with either flow_from_dataframe or flow_from_directory),
    do not specify the image size (ie, resize=False) or recolor the images (recolor=False). Resizing will cause issues, so 
    instead specify the image size in the flow_from_ function. Additionally, the flow_from_ functions automatically read in images
    as RGB (whereas cv2 reads them as BGR), so recoloring the images will have the opposite effect.
    """
    #Recolor
    if recolor:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        im = img
    #Adjust brightness
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    mean_val = np.mean(hsv[:,:,2])/255 #as a percentage of maximum pixel value
    if mean_val <= bright_threshold:
        h, s, v = cv2.split(hsv)
        lim = 255 - bright_value
        v[v > lim] = 255
        v[v <= lim] += bright_value
        final_hsv = cv2.merge((h, s, v))
        im = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    #Resize
    if resize:
        im = cv2.resize(im, image_size, interpolation = cv2.INTER_AREA)
    else:
        im = im
    return im


def preds_to_results(pred):
    """Converts model's raw predicted probs into information about predicted class"""
    # Initialize dictionary mappings
    i2tox = setup_dictionaries()
    # Top 5 class prediction
    # num_class = 15501
    # pred = pred.reshape(num_class)
    # pred_idx = np.argpartition(pred, -5)[-5: ]
    # Probabilities for top 5 preds
    # pred_prob = pred.reshape(num_class)[pred_idx]
    tox_pred_ix = np.where(pred > 0.5, 1, 0).item()

    # Map to get category labels
    pred_class = i2tox[tox_pred_ix]

    image_guess = pd.DataFrame({
    'class' : [pred_class],
    'prob' : [round(pred.item(), 3)],
    }).sort_values(by = 'prob', ascending=False)
    sorted_classes = [c for c in image_guess['class']]

    return image_guess, sorted_classes


def query_plant_info(categories, X=None,y=None):
    return None
    # # Locate row
    # rows = meta_data.loc[meta_data.category.isin(categories)]
    # # Extract info
    # category = [c for c in rows['category']]
    # scientific_name = [sc for sc in rows['scientific_name']]
    # family = [f for f in rows['family']]
    # genus = [g for g in rows['genus']]
    # species = [s for s in rows['species']]
    # images = []
    # # plot example image
    # if X is not None:
    #     for categ in [c for c in rows['category']]:
    #         img = X[np.where(y==categ)[0][0]]
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         images.append(img)
            
    #         #plt.imshow(img)
    #         #plt.axis('off')
    #         #plt.show()
    
    # return category, scientific_name, family, genus, species#, images


def setup_dictionaries():
    """Initialize dictionaries or metadata for use by other processes."""
    i2tox = {0: "Nontoxic", 1: "Toxic"}
    meta = pd.DataFrame({
    "class_id" : [0, 1, 2, 3],
    "slang" : ["Poison Oak", "Poison Ivy", "Poison Sumac", "Nontoxic"],
    "scientific_name" : ["Toxicodendron diversilobum / Toxicodendron pubescens", 
                        "Toxicodendron radicans / Toxicodendron rydbergii", 
                        "Toxicodendron vernix", "'Nontoxic'"],
    }) ##for use later...
    return i2tox

                         #------#
#------------------------# APP  #------------------------#
                         #------#
def main():
    """Run the app when the program is called."""
    with st.spinner('Loading model once...'):
        models = load_model()

    # INTRO
    st.write("""
            # 🌿 Rash Plant Classification 🌿
            ###  Helping to Identify Poison Oak, Ivy, and Sumac
            *DISCLAIMER:* This app is built on an image classification model which was trained to 86% test  
            accuracy. The best way to avoid poison oak, ivy, or sumac is to learn to identify them yourself. 
            
            #### Quick Facts:
            - **Eastern/Atlantic Poison Oak**: *Toxicodendron pubescens*. 3 leaflets, usually lobed, fuzzy when young but dull when mature. Can have light green to cream colored berries and small, light green flowers. Typically grows in shrub form.  
            - **Western/Pacific Poison Oak**: *Toxicodendron diversilobum*. 3 leaflets, usually lobed, both shiny and dull (often shiny). Can have White or cream colored berries and small, pink or green flowers. Can grow as vines or shrubs.
            - **Eastern/Atlantic Poison Ivy** & **Western/Pacific Poison Ivy**: *Toxicodendron radicans* & *Toxicodendron rydbergii*. 3 leafelts, either smooth (entire) or lobed, but never jagged (serrate). Can be shiny and fuzzy. May have white to cream colored berries and tiny white/green buds. Can grow as vines, shrubs, or ground cover.  
            - **Poison Sumac**: *Toxicodendron vernix.* 7-13 leaflets per single leaf, often very shiny and angled upward. May have white to green berries and greenish, drooping flowers. Grows as a shrub in wet, wooded areas. It is not as common as poison oak and ivy and tends to be found in the swampy areas of the southeastern and northeastern united states. 
            Scroll down for examples.  
            """
            )
    ip = "./streamlit/images/"
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.image(Image.open(ip+'plant-shine.jpg'), use_column_width=True, width=250)
    with col2:
        st.image(Image.open(ip+'leaf-margins.jpg'), use_column_width=True, width=250)
    with col3:
        st.image(Image.open(ip+'berries_flowers.jpg'), use_column_width=True, width=250)

    # leafs = [Image.open(paths[0])]
    # st.image(leafs, caption=[''],  use_column_width=False, width=250,)


    st.write(
            """
            ## Predict
            The model used for this project is a ResNet101v2 model pre-trained on ImageNet and then finetuned on
            a dataset scraped from iNaturalist.org. On the test data, it achieved an accuracy score of 86\% and ROC AUC score of 93\%.
            [For the training procedure see this example](https://www.kaggle.com/code/hanselliott/tpc-basicresnet/).
            """
            )

    #--------------------MAIN----------------------------------#
    # FILE UPLOADER
    file = st.file_uploader("Upload an image.", 
                            help="Supported filetypes: jpg/jpeg, png, heic (iPhone).") #type=["jpg", "png", "heic, "])
    st.set_option('deprecation.showfileUploaderEncoding', False)



    # PROCESS IMAGE AND PREDICT
    if file is None:
        st.text("Upload image for prediction.")
    else:
        bytes_data = file.read()
        filename = file.name
        # If file is in HEIC format (ie, if uploaded from iphone)
        if filename.split('.')[-1] in ['heic', 'HEIC', 'heif', 'HEIF']:
            heic_file = pillow_heif.read_heif(file)
            img = Image.frombytes(
                heic_file.mode,
                heic_file.size,
                heic_file.data
            )
        else:
            img = Image.open(file)
        st.write("### Your Image:")
        st.write("filename:", filename)
        st.image(img, width=400, use_column_width=False)
        out_table, out_text = upload_predict(img, models, image_size=(199, 199))
        st.write("# Prediction")
        st.write(out_table)
        st.write(out_text)



    # EXAMPLE IMAGES
    st.write("""
            
            ---
            #### Rash Plant Examples  
            Sources: [Leaf margins](https://biodiversity.utexas.edu/news/entry/leaves), [iNaturalist](https://www.inaturalist.org/) 
            """)
    # ex_imgs = [Image.open(paths[i+1]) for i in range(5)]
    paths = [ip+'east-pois-oak.jpg', ip+'east-pois-oak2.jpg',
            ip+'west-pois-oak.jpg', ip+'west-pois-oak2.jpg',
            ip+'east-pois-ivy.jpg', ip+'east-pois-ivy2.jpg',
            ip+'west-pois-ivy.jpg', ip+'west-pois-ivy2.jpg',
            ip+'pois-sumac.jpg', ip+'pois-sumac2.jpg',]

    ex_imgs = []
    for i in range(10):
        im = Image.open(paths[i])
        # im = im.resize((400, 400), Image.Resampling.LANCZOS) #cv2.resize(im, dsize=(400,400), interpolation=cv2.INTER_AREA)
        # im = im.crop((10, 10, 400, 400))
        ex_imgs.append(im)

    st.image(ex_imgs, caption=['Eastern Poison Oak','Eastern Poison Oak',
                                'Western Poison Oak', 'Western Poison Oak',
                                'Eastern Poison Ivy', 'Eastern Poison Ivy',
                                'Western Poison Ivy', 'Western Poison Ivy',
                                'Poison Sumac', 'Poison Sumac'
                                ],
                                use_column_width=False, width=300,)



    st.write("---")
    st.write("by [Hans Elliott](https://hans-elliott99.github.io/)")
    st.write("---")

if __name__=='__main__':
    main()