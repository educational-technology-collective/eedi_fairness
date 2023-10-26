import mlflow
import os
import pandas as pd

# Load student data which has columns UserId, Gender, DateOfBirth, PremiumPupil
student_df=pd.read_csv("./data/metadata/student_metadata_task_1_2.csv")

# Work on the arghosh entry requires switching to that sub-git
os.chdir("./entries/arghosh/")

from model_task_1_2 import *
from utils import *
from dataset_task_1_2 import *
from preprocessing_2 import *
import pickle
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download the best model to disk to experiment with
# best_run=mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name("arghosh_task_1").experiment_id,
#                    filter_string="tags.`best_run` LIKE '%'")
# location=mlflow.artifacts.download_artifacts(best_run['tags.best_run_uri'].squeeze())

# # load the model
# model_uri = f"{location}/model"
# loaded_model = mlflow.pytorch.load_model(model_uri).to(device)
# loaded_model.eval()


# now lets evaluate the model on the public validation dataset
evaluation_data={
    "trained":"../../data/train_data/train_task_1_2.csv",
    "public test":"../../data/test_data/test_public_answers_task_1.csv",
    "private test":"../../data/test_data/test_private_answers_task_1.csv"
}

for key in tqdm(evaluation_data.keys()):
    data=evaluation_data[key]

    # this is the location we will save the data to, note that it's a long
    # process to create features, so we should cache it here
    output_location=f"{key}.json"

    if os.path.exists(output_location):
        print("Cache found, loading featurized data")
        with open(output_location) as f:
            test_data=json.load(f)
    else:
        print(f"No cache found, featurizing data to {output_location}")
        # this needs to be converted into a JSON format for arghosh
        # keep in mind paths are related to the CWD
        # #'starter_kit/submission_templates/submission_task_1_2.csv'),
        test_data=featurize(TEST_DATA = pd.read_csv(data), 
                            output_location=output_location)

    # The 0.2 comes from trainer_task_1_2 and appears to be an old parameter
    # that the author hardcoded, previous param value was --valid-prob
    for d in test_data:
        d['valid_mask'] = [0 if np.random.rand(
        ) < 0.2 and ds else 1 for ds in d['test_mask']]

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

    assert model_parameters["is_dash"]==0, "Limitation, the dash value must be 0, e.g. no answer_meta at present as this requires additional preprocessing"

    test_dataset = LSTMDataset(test_data) #, answer_meta=answer_meta)
    collate_fn = lstm_collate() # is_dash=model_params['is_dash'] == 1)
    num_workers = int(os.cpu_count()/2)
    bs = 1 #previously was params.batch_size but looks like it is hard coded to 4
    test_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=False)


    model = LSTMModel(**model_parameters).to(device)
    model.eval()

    results=[] #collection of dataframes, one per user
    for batch in tqdm(test_loader):
        #evaluate this batch and make predictions
        y_pred = model(batch)

        #this is for prediction task #1 from the eedi 2020 neurips workshop
        #the labels are the true outcomes for the task in an ndarray
        #note that both y_pred[1] and target are 2 dimensional arrays,
        #as a single student might have many questions answered
        target = batch['labels'].squeeze().numpy()

        #keep track of how we have done, both labels and predictions
        #y_pred[0] has a tensor while y_pred[1] has the numpy predictions, just keep predictions
        
        #warning, this will not work if you are processing multiple users at once,
        #e.g. batch size !=1
        if bs==1:
            userid=batch['user_ids'][0]
            result={}
            result['predictions']=np.copy(y_pred[1])
            result['labels']=np.copy(target)
            result['userid']=userid
            # create an analysis dataframe for this user
            results.append(pd.DataFrame(result))
        else:
            print(f"Error, current script requires bs=1 to store data (per person evaluation)")

    # Generate the output file for analysis
    results_df= pd.concat(results)
    results_df.to_csv(f"../../{key}.results.csv")
