import mlflow
import os

experiment_name="arghosh_task_1"
try:
    experiment = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException as e:
    pass
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # change to arghosh working directory
    os.chdir("./entries/arghosh/")

    import trainer_task_1_2 as arghosh

    # set up model parameters
    model_parameters={'n_question':27613, 
            'n_user':118971, 
            'n_subject':389, 
            'n_quiz':17305, 
            'n_group':11844, 
            'hidden_dim':128,
            'q_dim': 32, 
            's_dim': 32,
            'task':"1",
            'dropout':0.25, 
            'default_dim':16, 
            'bidirectional': 1,
            'is_dash': 0}
    run_parameters={"seed":221,
         "deterministic":True,
         "benchmark":False,
         "valid_prob":0.2,
         "batch_size":2,
         "model":"lstm",
         "lr":1e-4,
         "weight_decay":1e-6,
         "max_epochs":100,
         "epoch_threshold":10}
    # run training
    arghosh.run(experiment.experiment_id,model_parameters,run_parameters)
