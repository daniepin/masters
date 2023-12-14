import glob
import json
import pandas as pd

path = r"/mnt/scratch/daniel/datasets/ukb_preprocessed/bids/**"

#files = glob.glob(path + "**", recursive=True)
#print(len(files))
def write_raw_file():
    files = [r'/mnt/scratch/daniel/datasets/ukb_preprocessed/ukb_img_path_data_ds.json', r'/mnt/scratch/daniel/datasets/ukb_preprocessed/uk_biobank_mri_dataset_cont_age.json', r'/mnt/scratch/daniel/datasets/ukb_preprocessed/ukb_TP_02_eid_age_sex.csv', r'/mnt/scratch/daniel/datasets/ukb_preprocessed/uk_biobank_mri_dataset.json', r'/mnt/scratch/daniel/datasets/ukb_preprocessed/ukb_data_participant_floatage_calculated.csv']
    with open(files[-2], 'r') as file:
        #print(json.load(file).keys())
        content = json.load(file)
        #df = pd.read_csv(file)
        #print(df.iloc[0])
        #print(file.read()) 

    print(content['normalization_coefficients'])
    all_files = content["train"] + content["val"] + content["test"]

    with open("uk_biobank_raw_info.csv", 'w') as file:
        file.write("patient_id,sex,age,session,image\n")
        for i, c in enumerate(all_files):
            file.write(f"{int(c['patient_id'])},{int(c['sex'])},{c['age']},{int(c['instance'])},{c['image']}\n")

def kfold_json(k: int = 5):
    with open("uk_biobank_raw_info.csv", 'r') as file:
        df = pd.read_csv(file)
        #print(df.head(5))
        total = len(df)


        df_sess3 = df[df["session"] == 3]
        df = df[df["session"] != 3]

        folds = {}

        for i in range(k):

            df_train = df.sample(frac=0.85, random_state=2023 + i)
            df_val = df_train.sample(frac=0.13, random_state=2023 + i)
            df_test = df.loc[~df.index.isin(df_train.index)]

            df_train = df_train.loc[~df_train.index.isin(df_val.index)]
            df_train = pd.concat([df_train, df_sess3])
            #train_uniques = df_train["patient_id"].unique()


            train_prc = len(df_train)*100/total
            val_prc = len(df_val)*100/total
            test_prc = len(df_test)*100/total

            print(f"train% = {train_prc}, val% = {val_prc}, test% = {test_prc}")

            print(len(df_train), len(df_val), len(df_test))
            print(len(df_train) + len(df_val) + len(df_test))
            print(total)

            train = []
            for row in df_train.itertuples(index=False):
                train.append(row._asdict())

            val = []
            for row in df_val.itertuples(index=False):
                val.append(row._asdict())

            test = []
            for row in df_test.itertuples(index=False):
                test.append(row._asdict())
            
            folds[f"fold_{i}"] = {
                "train": train,
                "val": val,
                "test": test
            }
    
    with open(f"{k}_fold_split.json", 'w') as out:
        json.dump(folds, out)

#write_raw_file()
kfold_json(5)