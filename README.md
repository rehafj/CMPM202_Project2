# CMPM202 classProject 
## About: 
This project includes two parts,  a recurrent neural network, and an autoencoder modified for class purposes.
please see the descriptions below for both parts. 

## part1 - RNN and generating text 
In part one of the project, we implemented a recurrent neural network using tensorflow to generate text. We followed and modified the tutorial found at https://www.tensorflow.org/tutorials/sequences/text_generation

We ran the code with text pieces from various resources. including books and musicals
Sample generated text includes: 

* Lord of the rings book
* musicals ( Hamilton and wicked), 
* and a game of thrones. 

All text is formatted in a way that fits its category; as an example, we formatted Hamilton and Wicked lyrics to include characters between brackets and increased spacing between segments. 


### running the code: 

To run part one of the project, please install the required dependencies mainly *tensorflow*(1.2), *numpy* and *nightlye* for compatibility issues and traverse into part one of the directory. 

```
install pip  install -q tf-nightlye
```

to run all the samples mentioned above, run the **_autorun.py_** script without any parameters 

### Sample output: 
**Lord of the rings**

with Epoch 3, nonsical text was produced
> It waste nongey flotly. It mom the astre, ma, cronden hians, on yay the, in th wack tht de, butors wrouthed been res afordote, op dhilbechnm' anbi par ftam amd turbe high to ho co mooks wouthe llatither and he dipep lofen. 

with Epoch 5, more training: 

>The hope of the darkness were steader. 
Sam darkness. 
The fingents of the Loddless of the meapon of the open southwards. But we is still because there, or -lack

**Hamilton**



**Hamilton and Wicked**


**Game of thrones**


## part2 - Autoencoder and un-pixelating images 

In part Two of the project, we altred an autoencoder provided in class by Manu Tomas, @manumathewthomas. Our goal was to train the model to smooth out pixel images. 
We modified a pixellating python script and created a pixilated data set  to use along with a clear data set. 

The data set includes more cat:cat: images than standard images thus the model was better trained against cat images as the results would suggest. We also discovered that photos with a definite object in its focus result in a better un-pixelized photo, while pictures that include a scenery or background object were still pixelated in comparison. 


### running the code: 

If you would like to train the model with a different set of images, please nest them appropriately under dataset/train, dataset/val, and dataset/test ( both pixelated and clear images should be provided), then run``` bethsAutotester.py ```

To use it against our trained model, please run the script: 

``` ... ```

### results:

Before training the model ( using the focus based model ) resulted in 

 <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/pretraining.png" height ="200" width="200"> 

### test 1 and 2, defined below
Running the modeThis test was a CPU based test and training the model for 8 hours. 

pixelated input images: 

<p float="left">
    <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test2/16p.jpg" height="200" width="200" /> 

  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test2/7p.jpg" height="200" width="200" />
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test2/0p.jpg" height="200" width="200" /> 
</p>


output resultS: 
surprisingly, results 1 showed less pixelated images than test 2

Test 1  | Test 2
:-------------------------:|:-------------------------:
9k iterations, 32 filters, 3 batch| 9k iterations, 62 filters, 5 batch 
![](sample_results/test1/0depixelated.jpg)  |  ![](sample_results/test2/16dp.jpg)
![](sample_results/test1/12dp.jpg)   | ![](sample_results/test2/7dp.jpg)
![](sample_results/test1/6dp.jpg) |  ![](sample_results/test2/odp.jpg)

### test 0 
Test 0: 100k iterations, 32 filters and 3 batch size
Original image, pixelated input, de-pixelized output
<p float="left">
    <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/cat%20sample/2.jpg" height="200" width="200" /> 

  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/cat%20sample/input_p.jpg" height="200" width="200" />
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/cat%20sample/output_d.jpg" height="200" width="200" /> 
</p>

<p float="left">
    <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/4116.jpg" height="200" width="200" /> 
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/1p.jpg" height="200" width="200" />
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/1dp.jpg" height="200" width="200" /> 
</p>



<p float="left">
    <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/4100.jpg" height="200" width="200" /> 
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/6p.jpg" height="200" width="200" />
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/6dp.jpg" height="200" width="200" /> 
</p>


<p float="left">
    <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/4116.jpg" height="200" width="200" /> 
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/12p.jpg" height="200" width="200" />
  <img src="https://github.com/rj-90/CMPM202_Project2/blob/master/sample_results/test0/12dp.jpg" height="200" width="200" /> 
</p>











## Credit and resources: 
 Text gen tutorial: https://www.tensorflow.org/tutorials/sequences/text_generation
 autoencoder base code by @manumathewthomas:https://github.com/manumathewthomas
 Original pixelation code:  https://gist.github.com/danyshaanan/6754465
 
### team members: 
Elisabeth Oliver @bluestar514 and Rehaf jammaz


