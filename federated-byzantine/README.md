# Federated Learning using Pytorch

## Running the code

```bash
python3 run.py setup        # Setup directories
python3 run.py getdata      # Download MNIST dataset
python3 run.py splitdata 5  # Split the dataset into 5 pieces for 5 nodes
python3 run.py worker <id> <port>  # Launch a worker
python3 run.py createmodel  # Generate a randomly initialised model
python3 run.py train <C> <addr0> <addr1> ... # Federated training by choosing C nodes randomly from the <addr0> <addr1> ... list
python3 run.py collect <id0>:<size0> <id1>:<size1> ... # Collect the models trained by different nodes and apply Federated Averaging
# To get the ids, note down the address printed during training and get the ids from the worker spawning step.
# To get the sizes, check the output printed during training.
python3 run.py clean # To clean the previously trained worker models so that we have a fresh outputs directory for the next run.
```