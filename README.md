# AI Music to Image Generator

**Ongoing project** (excuse the mess)

The idea of this project is to use an AI approach to generate images from my music using a combination of CNN and Diffusion Models. The models will be trained (or fine-tuned) on the association of my music with my 3D renders in order to have image generation that follows the aesthetic vision of my artistic project. 

Outline of the project:
1. Audio file gets converted into Mel Spectrogram 
2. The spectrogram is fed into a CNN model for feature extraction
3. Unsupervised clustering of the images (K-Means or DBSCAN like) to group music by “vibe”
4. My 3D renders will be associated to each Spectrogram following the “vibe” clustering
5. Now features of Spectrogram - 3D renders relation will be used to personalise LoRA of a Diffusion Model

Approach:
1. Refer to my other repository for Audio to Spectrogram conversion (https://github.com/pcaprioglio/Audio-to-Spectrogram-app)
2. Here I will try different approaches. The unsupervised approach could be done by using pre-trained CNNs as in my other projects (https://github.com/pcaprioglio/Unsupervised-DL-image-classifier-routine) and then proceed with clustering using K-means or DBSCN. 
Alternatively, I sketched a simple custom CNN to be trained on this music dataset: http://millionsongdataset.com/. Ideally, I would like to perform the training actually on electronic subgenres using big catalogues like Beatport since fits more with the subtle differences of my music style. 
I’m also considering using this model for feature extraction: https://github.com/LAION-AI/CLAP

Step 3 and 4 a trivial. 

5. This is core part for the actual aesthetic output. Probably going for fine-tuning this: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
