# snn_conversion_experiments_2

Converting ESCAPE_NET CNN to SNN using Conversion and Surrogate Gradient Descent Methods

# Goal and Problem Definition: (Yet to be completed)
The goal of this summer's work was to confirm or deny the validity of a Spiking Neural Network (SNN) that is converted from a trained Artificial Neural Network (ANN), in this case the ESCAPE-NET convolutional neural network that was trained and published [1]. The validity of an ESCAPE_NET SNN is based on achieving the following metrics:
* Accuracy: 
  * The accuracy of the SNN should be as close to that of the ANN as possible.
  * *How do we determine the balance between inference latency/energy savings to the desired accuracy we want to achieve* (Yet to be rigourously addressed)
    * Given that the error in the ANN itself is quite large, the increase in error of the SNN is not that concerning (i.e. an accuracy drop from 95 to 90 is a lot worse than from 73 to 70) 
    * It's hard to gauge a minimum required accuracy the network needs to achieve given how poor the original model is. ESCAPE-NET achieves 73.76% accuracy on a 7865 image test set.
* Inference time:
  * 
* Energy Efficiency:
  * Most important metrics to consider given the final use case in an implantable device; needs to consume as little power as possible
  * See next section for the energy model used to guide and validate results

# Energy Model for CNN and SNN (Maybe move this to separate readme as well)
(Yet to be added)

# Experiment Details: (move all of this off the main readme eventually)
## ESCAPE_NET Model to be Converted:
Original model trained and published by Ryan  
![image](https://user-images.githubusercontent.com/58120600/188368954-3993c46f-29be-47c8-99d2-aa4c7674ade4.png)
## Input Encoding:
  * Only focused on the most widely used input input encoding schemes, rate-coding [6] due to its simplicity and available resoruces
  * Rate-coding uses each input pixel value as a firing rate and converts the pixel into a Poisson spike train with the firing rate
  * **Figure 1. Original Input Image**
    * ![image](https://user-images.githubusercontent.com/58120600/188369277-cc5bc6bf-42d5-4a7c-b874-3f9f87492c36.png)

  * **Figure 2. Sample Poisson Generated Input Spikes**
    * ![image](https://user-images.githubusercontent.com/58120600/188369203-fc561fed-1852-43f9-8b8a-da3ff677041a.png)
## Experiment 1:
The ANN to SNN conversion technique was utilized as a simple way to obtain an SNN without having to deal with the issues of backpropagation of a non-differentiable function such as the binary spikes in an SNN. Deep SNNs trained with conversion method showed to have the best results [2]
* **Details from Paper:**
  * The basic idea behind ANN to SNN conversion is to copy the weights from the trained ANN and set the firing threshold for each layer as the max input received in that layer. The inputs to all neurons are then integrated across the simulation time and spikes are propagated between neurons when the membrane potential surpasses the threshold. 
  * Additionally, this particular conversion method averages 2 different neuron reset mechanisms (to zero and subtraction). There are many available conversion tools and I am currently using one called SNNtoolbox by Rueckauer et al. [3][4]
* **Approach:**
  * Retrain Ryan's RAT4 model in TensorFlow 
  * input the trained model into the conversion pipeline
  * This technique allows us to change as little as possible to the current pipeline since we can simply reuse the weights of the CNN
* **Results:**
  * through initial experiments it was found that by using this method, the converted SNN would require a lot of simulation timesteps to converge to ANN performance (see Table 1 in RESULTS for details). From the results it was clear that neither energy or inference times were being improved with this conversion method

## Experiment 2:
To deal with the issue of long simulation times of a converted SNN, I worked towards implementing the methods described in the paper, 'Hybrid Conversion and Spike Timing Dependent Backpropagation' [5].
* **Details from Paper:**
  * Convert the trained ANN to SNN similar to prior experiment
	* Take converted SNN and use its weights and thresholds as an initialization for spike-based backpropagation
  * Perform incremental spike-timing dependant backpropagation (STDB) on the initialized network. The backpropagation is done using a surrogate gradient function to alleviate the non-differentiability of the binary spike. 
	* Baseline results from paper: 
    * Network performs at 10-25x fewer timesteps and achieve similar accuracy
    * Training method converges in few epochs which was a huge deterring factor previously to training an SNN using backpropagation for many timesteps
* **Approach:**
  * Transition the codebase and training scripts from tensorflow to pytorch
  * Make minor changes to the structure and retrain ESCAPE_NET. ANN to SNN conversion shows to have the best results when:
		* AVG pool is used instead of MAX
		* No Batch Normalization or Bias; dropouts should be used for regularization
  * Input the trained model into the conversion pipeline

# Results of the 2nd Experiment
**Table 1. Overall Results Evaluated on 7835 Test Images**  

| Model             | Accuracy                    | Normalized Energy Consumption | Inference Latency |
| ----------------- | --------------------------- | ----------------------------- | ----------------- |
| Original ANN      | 0.738                       | 1                             |                   |
| Modified ANN      | 0.727                       | N/A                           | N/A               |
| ANN to SNN only   | 0.713 (best, 300 timesteps) | N/A                           | N/A               |
| ANN to SNN + STDB | 0.699 (best, 28 timesteps)  | 0.0194174757 (51.5x saving)   |                   |

**Table 2. ANN to SNN only Results**  
| Timesteps | ANN Accuracy | ANN to SNN only |
| --------- | ------------ | --------------- |
| 50        | 0.727        | 0.422           |
| 75        | 0.727        | 0.588           |
| 100       | 0.727        | 0.6623          |
| 200       | 0.727        | 0.7110          |
| 300       | 0.727        | 0.7130          |

![image](https://user-images.githubusercontent.com/58120600/188371614-2d7d5f3b-30f4-4950-859f-3539b5bb00a8.png)  
* As expected, longer simulation means better performance for the converted only SNN

**Table 3. ANN to SNN + STDB Results:**
| ANN Accuracy | Timesteps | Leak | SNN only Accuracy | SNN STDB Accuracy | Best Epoch |
| ------------ | --------- | ---- | ----------------- | ----------------- | ---------- |
| 0.727        | 28        | 1    | 0.283             | 0.699             | 10         |
| 0.727        | 25        | 1    | 0.279             | 0.696             | 11         |
| 0.727        | 35        | 1    | 0.306             | 0.695             | 12         |
| 0.727        | 25        | 1    | 0.279             | 0.695             | 9          |
| 0.727        | 50        | 1    | 0.422             | 0.69              | 2          |
| 0.727        | 75        | 1    | 0.588             | 0.681             | 7          |
| 0.727        | 20        | 1    | 0.277             | 0.68              | 12         |
| 0.727        | 15        | 1    | 0.28              | 0.676             | 12         |
| 0.727        | 100       | 1    | 0.662             | 0.669             | 7          |
| 0.727        | 50        | 1    | 0.422             | 0.666             | 3          |
| 0.727        | 10        | 1    | 0.273             | 0.609             | 12         |
| 0.727        | 5         | 1    | 0.274             | 0.274             | 1          |

**Figure 3. Training Curves for SNN STDB with 28 Timesteps**  
<img src="https://user-images.githubusercontent.com/58120600/188372003-8f3d1474-cf3f-4b87-9043-77c44d42feb0.png" width="450" height="320">
<img src="https://user-images.githubusercontent.com/58120600/188372012-171fa06f-7ece-487f-ab98-fd94f475d916.png" width="450" height="320">

##  Analyzing the Spiking Nature of the SNN + STDB and Evaluating Overall Energy Savings  
(The figures below are created for a single sample that has been simulated for 28 timesteps)

Layers Summary:  
| Layer Number | Name    | Neurons |
| ------------ | ------- | ------- |
| 0            | Conv1   | 358400  |
| 3            | Conv2   | 89600   |
| 6            | Conv3   | 22400   |
| 8            | Linear1 | 256     |
| 10           | Linear2 | 3       |

**Figure 4. Spiketrains for every neuron in each layer plotted over the simulation duration**  (click images to zoom)  
<img src="https://user-images.githubusercontent.com/58120600/188373354-866b4dd2-cc6e-4f1b-a0a6-ec615e92b073.png" width=20% height=20%><img src="https://user-images.githubusercontent.com/58120600/188373378-898c3533-7b1d-42d7-a129-8a1ce99fca1c.png" width=20% height=20%><img src="https://user-images.githubusercontent.com/58120600/188373394-4659c89c-5d8a-4a8d-843e-d614ab5ec754.png" width=20% height=20%><img src="https://user-images.githubusercontent.com/58120600/188373406-20fdb551-3d22-4154-a39a-6f39869a71cb.png" width=20% height=20%><img src="https://user-images.githubusercontent.com/58120600/188373417-16d65d4a-5612-493d-a500-dc8460e1e184.png" width=20% height=20%>
* *Note how the sparsity of spikes increases as you get deeper into the network.*

**Figure 5. Histogram of the number of spikes over the simulation duration for every neuron in each layer**  
<img src="https://user-images.githubusercontent.com/58120600/188377359-e7033a3c-38c8-484e-b6aa-4a22e5042396.png" width=25% height=25%><img src="https://user-images.githubusercontent.com/58120600/188374211-ef9143e9-d878-4495-a32b-40c6f7dbf716.png" width=25% height=25%><img src="https://user-images.githubusercontent.com/58120600/188374281-9327afb2-cb39-4961-9ab6-b4e3d3daf47e.png" width=25% height=25%><img src="https://user-images.githubusercontent.com/58120600/188374383-ae76d36c-b4c2-4c8a-bf46-405b17bacd22.png" width=25% height=25%>
* Note how around 90% of the neurons in each layer do not spike once
* This means that a majority of the pixels in the original spatio-temporal signatures of the RAT dataset carry very little information
* *This shows that this type of data and PNS application has great synergy with SNNs*
 
**Figure 6. Computations in ANN Layer vs Spikes in SNN Layer**  
![image](https://user-images.githubusercontent.com/58120600/188378549-f4fd1f60-ceb2-4cdc-ae56-d87df71e424b.png)  
**Figure 7. Normalized Energy Consumption in ANN vs. SNN**  
![image](https://user-images.githubusercontent.com/58120600/188378726-14464b55-b583-4a76-8f16-f95da5794f44.png)  
**Figure 8. Energy Consumption throughout Simulation**  
![image](https://user-images.githubusercontent.com/58120600/188378993-96a51c04-a3f3-441c-aa09-1d68959b8e55.png)  

# How to Use Repository

