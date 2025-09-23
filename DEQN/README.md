## What is in the folder

This folder contains the source code for running a common solution routine for the solutions of the climate-economy models presented in the paper. 

* Each directory has the model name XXX and contains the following scripts:

  * Definitions (derived quantities)
  * Dynamics (state transitions)
  * Equations (equilibrium conditions)
  * Hooks (things to be done after each episode - e.g. plotting)
  * Variables (description of constants and variables)

* Directory config contains folders specifying neural net architecture and hyperparameters:

  * How the model is simulated can be found inside `config/run/*.yaml`. The latter file needs to contain the following variables:

```
N_sim_batch: 1000
N_episode_length: 30
N_epochs_per_episode: 1
N_minibatch_size: 500
N_episodes: 500000
sorted_within_batch: false
```

Important to note: N_sim_batch is the number of trajectories simulated in parallel, so e.g. 1000 episodes are simulated in parallel. Each (parallel) episode is 30 long in this case, so altogether 30000 states are drawn at each episode iteration. These are then batched into N_minibatch_size for gradient steps (so in this case 60 minibatches are drawn of size 500, for each episode). If `sorted_within_batch` is set, then the batches will try to be contiguous trajectory elements. This should in general entail that N_episode_length is a multiple of N_minibatch_size and each minibatch then is a contiguous fragment of one trajectory.

  * The architecture of the neural net is specified in `config/net/*.yaml` There is a possibility to add a 'dropout_rate' to a layer configuration. In this case a Dropout layer is added before that layer.
The Dropout layer is only active during the 'epoch' run (calculating losses & doing a gradient step based on that) and is not active during the episode generation (i.e. simulation) and when running Hooks (e.g. plotting policies).

```
layers:
  - hidden:
     units: 100
     type: dense
     activation: relu
     init_scale: 0.1
     dropout_rate: 0.05
  - hidden:
     units: 50
     type: dense
     activation: relu
     init_scale: 0.1
     dropout_rate: 0.05
  - output:
     type: dense
     activation: linear
     init_scale: 0.1
net_initializer_mode: fan_avg
net_initializer_distribution: truncated_normal
```
There is also the possibility to choose the distribution (uniform, normal, truncated_normal) and mode of the neural network initializer (eg. fan_in, fan_out or fan_avg). Together with the scale parameter, this should cover https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform as well. See the exact details at: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling.

Batch-normalization can be done before a given layer, by adding a `batch_normalize` entry to the layer, like:

```
layers:
  - hidden:
     units: 100
     type: dense
     activation: relu
     init_scale: 0.1
     batch_normalize:
       momentum: 0.99
```
The parameters of this field are passed on directly to https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

  * There exists the possibility to choose the optimizer `config/optimizer/*.yaml` - just specify it in the optimizer name (as listed in https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) in config/optimizer/yourconfigfile.yaml.

```
optimizer: Adam
learning_rate: 0.00001
clipvalue: 1.0
```

* The solution results for all the models (in *.csv format) as well as pretrained models for a replication are stored in the folder [runs](runs).

## How to run a specific model
To start the computation from scratch, the only thing to change in the config file (`config/config.yaml`) is the name of the model, while leaving the other entries untouched :

```
MODEL_NAME:  XXX
```
XXX stands for the specific name of the model, that are presented below.

Thereafter, make sure you are at the root directory of DEQN (e.g., ~/DEQN), and
execute:

```
python run_deepnet.py
```

## List of the model names for running from scratch (what to change in the config file):

**cdice_3sr_pd_opt:**
The IAM with 3 reservoirs with 3SR calibration under PD, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pd_opt
```

---

**cdice_3sr_pi_bau:**
The IAM with 3 reservoirs with 3SR calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_bau
```

---

**cdice_3sr_pi_ccs_opt:**
The IAM with 3 reservoirs and Carbon Capture and Storage (CCS) Technology with 3SR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_ccs_opt
```

---

**cdice_3sr_pi_climber_bau:**
The IAM with 3 reservoirs with 3SR-CLIMBER2-LPJ calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_climber_bau
```

---

**cdice_3sr_pi_climber_opt:**
The IAM with 3 reservoirs with 3SR-CLIMBER2-LPJ calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_climber_opt
```

---

**cdice_3sr_pi_highdam_opt:**
The IAM with 3 reservoirs with high damages and 3SR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_highdam_opt
```

---

**cdice_3sr_pi_mesmo_bau:**
The IAM with 3 reservoirs with 3SR-MESMO calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_mesmo_bau
```

---

**cdice_3sr_pi_mesmo_opt:**
The IAM with 3 reservoirs with 3SR-MESMO calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_mesmo_opt
```

---

**cdice_3sr_pi_opt:**
The IAM with 3 reservoirs with 3SR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_opt
```

---

**cdice_3sr_pi_psi05_opt:**
The IAM with 3 reservoirs with 3SR calibration and psi=0.5 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_psi05_opt
```

---

**cdice_3sr_pi_psi2_opt:**
The IAM with 3 reservoirs with 3SR calibration and psi=2 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_3sr_pi_psi2_opt
```

---

**cdice_4pr_pd_opt:**
The IAM with 4 reservoirs with 4PR calibration under PD, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pd_opt
```

---

**cdice_4pr_pi_bau:**
The IAM with 4 reservoirs with 4PR calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_bau
```

---

**cdice_4pr_pi_ccs_opt:**
The IAM with 4 reservoirs and Carbon Capture and Storage (CCS) Technology with 4PR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_ccs_opt
```

---

**cdice_4pr_pi_climber_bau:**
The IAM with 4 reservoirs with 4PR-CLIMBER2-LPJ calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_climber_bau
```

---

**cdice_4pr_pi_climber_opt:**
The IAM with 4 reservoirs with 4PR-CLIMBER2-LPJ calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_climber_opt
```

---

**cdice_4pr_pi_highdam_opt:**
The IAM with 4 reservoirs with high damages and 4PR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_highdam_opt
```

---

**cdice_4pr_pi_mesmo_bau:**
The IAM with 4 reservoirs with 4PR-MESMO calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_mesmo_bau
```

---

**cdice_4pr_pi_mesmo_opt:**
The IAM with 4 reservoirs with 4PR-MESMO calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_mesmo_opt
```

---

**cdice_4pr_pi_opt:**
The IAM with 4 reservoirs with 4PR calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_opt
```

---

**cdice_4pr_pi_psi05_opt:**
The IAM with 4 reservoirs with 4PR calibration and psi=0.5 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_psi05_opt
```

---

**cdice_4pr_pi_psi2_opt:**
The IAM with 4 reservoirs with 4PR calibration and psi=2 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4pr_pi_psi2_opt
```

---

**cdice_4prx_pi_bau:**
The IAM with 4 reservoirs with 4PR-X calibration under PI, BAU solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_bau
```

---

**cdice_4prx_pi_ccs_opt:**
The IAM with 4 reservoirs and Carbon Capture and Storage (CCS) Technology with 4PR-X calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_ccs_opt
```

---

**cdice_4prx_pi_highdam_opt:**
The IAM with 4 reservoirs with high damages and 4PR-X calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_highdam_opt
```

---

**cdice_4prx_pi_opt:**
The IAM with 4 reservoirs with 4PR-X calibration under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_opt
```

---

**cdice_4prx_pi_psi05_opt:**
The IAM with 4 reservoirs with 4PR-X calibration and psi=0.5 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_psi05_opt
```

---

**cdice_4prx_pi_psi2_opt:**
The IAM with 4 reservoirs with 4PR-X calibration and psi=2 under PI, optimal solution.

For running the model from scratch:
```
MODEL_NAME:  cdice_4prx_pi_psi2_opt
```

## Monitoring

Monitoring of the running solution can be done via Tensorboard, pointing it to the hydra.run.dir directory. Diagnostic information (e.g. loaded config values, current iteration, etc...) is printed also to stdout.


## How to analyze the pre-computed solutions

To analyze the the raw results presentended in the article, you need to
perform two steps that are outlined in the section below for each model. Please note that postprocessing scripts differ, depending on the type of the model. Specifically BAU solution, optimal solution and solution with CCS technology require different prostrpocessing routines. All necessary steps for each model are provided below.


## List of the postprocessing routines for models

**cdice_3sr_pd_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pd_opt/3sr_pd_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_bau:**
```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_bau/3sr_pi_bau

python post_process_3sr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_ccs_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_ccs_opt/3sr_pi_ccs_opt

python post_process_3sr_ccs.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_climber_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_climber_bau/3sr_pi_climber_bau

python post_process_3sr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_climber_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_climber_opt/3sr_pi_climber_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

* IMPORTANT!

Due to the large size of the pretrained solutions, solutions of the model **cdice_3sr_pi_climber_opt** are sotred as *.tar.gz archive:

To unarchive respective solutions please follow:

```tar -zxvf 3sr_pi_climber_opt.tar.gz```

---

**cdice_3sr_pi_highdam_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_highdam_opt/3sr_pi_highdam_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_mesmo_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_mesmo_bau/3sr_pi_mesmo_bau

python post_process_3sr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_mesmo_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_mesmo_opt/3sr_pi_mesmo_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DI
```

---

**cdice_3sr_pi_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_opt/3sr_pi_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_psi05_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_psi05_opt/3sr_pi_psi05_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_3sr_pi_psi2_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_3sr_pi_psi2_opt/3sr_pi_psi2_opt

python post_process_3sr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pd_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pd_opt/4pr_pd_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_bau/4pr_pi_bau

python post_process_4pr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_ccs_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_ccs_opt/4pr_pi_ccs_opt

python post_process_4pr_ccs.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_climber_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_climber_bau/4pr_pi_climber_bau

python post_process_4pr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_climber_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_climber_opt/4pr_pi_climber_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
* IMPORTANT!

Due to the large size of the pretrained solutions, solutions of the model **cdice_4pr_pi_climber_opt** are sotred as *.tar.gz archive:

To unarchive respective solutions please follow:

```tar -zxvf 4pr_pi_climber_opt.tar.gz```

---

**cdice_4pr_pi_highdam_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_highdam_opt/4pr_pi_highdam_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_mesmo_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_mesmo_bau/4pr_pi_mesmo_bau

python post_process_4pr_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_mesmo_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_mesmo_opt/4pr_pi_mesmo_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
* IMPORTANT!

Due to the large size of the pretrained solutions, solutions of the model **cdice_4pr_pi_mesmo_opt** are sotred as *.tar.gz archive:

To unarchive respective solutions please follow:

```tar -zxvf 4pr_pi_mesmo_opt.tar.gz```

---

**cdice_4pr_pi_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_opt/4pr_pi_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_psi05_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_psi05_opt/4pr_pi_psi05_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4pr_pi_psi2_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4pr_pi_psi2_opt/4pr_pi_psi2_opt

python post_process_4pr.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_bau:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_bau/4prx_pi_bau

python post_process_4prx_bau.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_ccs_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_ccs_opt/4prx_pi_ccs_opt

python post_process_4prx_ccs.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_highdam_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_highdam_opt/4prx_pi_highdam_opt

python post_process_4prx.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_opt/4prx_pi_opt

python post_process_4prx.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_psi05_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_psi05_opt/4prx_pi_psi05_opt

python post_process_4prx.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```

---

**cdice_4prx_pi_psi2_opt:**

```
export USE_CONFIG_FROM_RUN_DIR=<PATH_TO_THE_FOLDER>/Replication_Building_Emulators/DEQN/runs/cdice_4prx_pi_psi2_opt/4prx_pi_psi2_opt

python post_process_4prx.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR
```
* IMPORTANT!

Due to the large size of the pretrained solutions, solutions of the model **cdice_4prx_pi_psi2_opt** are sotred as *.tar.gz archive:

To unarchive respective solutions please follow:

```tar -zxvf 4prx_pi_psi2_opt.tar.gz```

For more details about postprocessing results please refer to [runs](runs).

## IMPORTANT!

Due to the large size of the pretrained solutions, solutions of the following models are sotred as *.tar.gz archive:

- **cdice_3sr_pi_climber_opt**
- **cdice_4pr_pi_climber_opt**
- **cdice_4pr_pi_mesmo_opt**
- **cdice_4prx_pi_psi2_opt**

* To unarchive respective solutions please follow:

```tar -zxvf XXX.tar.gz```
where XXX stands for the name of the archive.










