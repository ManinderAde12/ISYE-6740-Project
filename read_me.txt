This describes image_oprganizer.py & resample_images.py

The idea is to have a data pipeline.

image_organizer.py walks the folders of chest xray images and creates a pandas df
that organizes the image file names by patient number and type (normal, bacteria, virus)
instead of storing the file contents it just stores the filenames which will later be used 
import data for processing. image_organizer also distributes the patients across train,
test and validate sets.
Data is exported to csv (for readibility) and json (to preserve df structure)

resample_images.py oversamples normal and virus images in each set (train.test.validate) to
have the same number of images as bacteria in that set, to prevent model bias. Data is again
exported to csv (for readibility) and json (to preserve df structure)

My idea is to stick to standardize these intermediate files for modeling. That way if we want 
to try different distributions (such as cross validation) or different oversampling methods,
we can then just point the models to the different inputs.

Not sure if all modeling will be based on PCA for dimension reduction.  If so I would be happy to 
code another processor to read in the output from above and generate a file that had the image data 
projected to the first 50 eigenvectors.  I think this could be valuable as I could run that on my 
beefy desktop and post the output, to avoid having to recompute PCA with each model run.

