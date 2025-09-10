
import globus_sdk
from globus_sdk.scopes import GCSCollectionScopeBuilder, TransferScopes
import pandas as pd
import os, shutil, time, argparse
from tqdm import tqdm

def run(args):
    CLIENT_ID = args.client_id
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    auth_client.oauth2_start_flow()

    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    auth_code = input("Please enter the code here: ").strip()

    token_response = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]
    TRANSFER_TOKEN = globus_transfer_data["access_token"]

    authorizer = globus_sdk.AccessTokenAuthorizer(TRANSFER_TOKEN)
    transfer_client = globus_sdk.TransferClient(authorizer=authorizer)

    id_source = args.id_source
    id_target = args.id_target
    transfer_client.endpoint_autoactivate(id_source)
    transfer_client.endpoint_autoactivate(id_target)

    f = open(args.id_dict, "r", encoding="utf-8")
    ie_to_rep = json.load(f)

    for i, ie in enumerate(tqdm(ie_to_rep.keys(), desc="Processing IE PIDs")):
        for rep in ie_to_rep[ie]:
            output_path = os.path.join(args.save_path, f"{ie}_{rep}")
            if os.path.isdir(output_path):
                    continue
            # if os.path.exists(output_path):
            #     shutil.rmtree(output_path)
            os.makedirs(output_path)

            input_path = os.path.join(args.globus_data_path, ie, rep)

            task_data = globus_sdk.TransferData(
                transfer_client, id_source, id_target, label=f"{ie}_{rep}"
            )
            task_data.add_item(
                input_path,   
                output_path,  
                recursive=True
            )

            task_doc = transfer_client.submit_transfer(task_data)
            task_id = task_doc["task_id"]
            print(f"\n Submitted transfer for {ie}/{rep}, task_id={task_id}")

            while True:
                task = transfer_client.get_task(task_id)
                status = task["status"]

                files = task.get("files", 0)
                files_transferred = task.get("files_transferred", 0)
                bytes_transferred = task.get("bytes_transferred", 0)
                bytes_total = task.get("bytes", 0)

                pct_files = (files_transferred / files * 100) if files else 0
                pct_bytes = (bytes_transferred / bytes_total * 100) if bytes_total else 0

                print(f"{ie}/{rep} | Status: {status}")
                print(f"   Files: {files_transferred}/{files} ({pct_files:.1f}%)")
                print(f"   Data: {bytes_transferred/1e6:.1f} MB / {bytes_total/1e6:.1f} MB ({pct_bytes:.1f}%)")

                if status in ["SUCCEEDED", "FAILED"]:
                    print(f" Transfer for {ie}/{rep} finished with status: {status}\n")
                    break

                time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, help='Globus client ID')
    parser.add_argument("--id_source", type=str, help='UUID of the source collection (visible in globus)')
    parser.add_argument("--id_target", type=str, help='UUID of the target collection')
    parser.add_argument("--id_dict", type=str, default="data/ie_dict.json", help=" 'json that stores the manuscript IDs (globus)'")
    parser.add_argument("--save_path", type=str, default="magister_dixit", help='path where the images will be saved')
    parser.add_argument("--globus_data_path", type=str, default="/DATASET_1", help='path of the dataset in globus')

    args = parser.parse_args()
    run(args)
