import json
import boto3
from s3_helper import S3Client
import sys
from datetime import datetime

old_sg_outdir = "data/not_validated_scene_graphs"
old_framedir = "data/not_validated_frames"

sg_outdir = "data/validated_scene_graphs"

curr_time = str(datetime.now().strftime("%m%d_%H%M"))

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """This is a sample Annotation Consolidation Lambda for custom labeling jobs. It takes all worker responses for the
    item to be labeled, and output a consolidated annotation.
    Parameters
    ----------
    event: dict, required
        Content of an example event
        {
            "version": "2018-10-16",
            "labelingJobArn": <labelingJobArn>,
            "labelCategories": [<string>],  # If you created labeling job using aws console, labelCategories will be null
            "labelAttributeName": <string>,
            "roleArn" : "string",
            "payload": {
                "s3Uri": <string>
            }
            "outputConfig":"s3://<consolidated_output configured for labeling job>"
         }
        Content of payload.s3Uri
        [
            {
                "datasetObjectId": <string>,
                "dataObject": {
                    "s3Uri": <string>,
                    "content": <string>
                },
                "annotations": [{
                    "workerId": <string>,
                    "annotationData": {
                        "content": <string>,
                        "s3Uri": <string>
                    }
               }]
            }
        ]
        As SageMaker product evolves, content of event object & payload.s3Uri will change. For a latest version refer following URL
        Event doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html
    context: object, required
        Lambda Context runtime methods and attributes
        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    Returns
    ------
    consolidated_output: dict
        AnnotationConsolidation
        [
           {
                "datasetObjectId": <string>,
                "consolidatedAnnotation": {
                    "content": {
                        "<labelattributename>": {
                            # ... label content
                        }
                    }
                }
            }
        ]
        Return doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html
    """

    labeling_job_arn = event["labelingJobArn"]
    label_attribute_name = event["labelAttributeName"]

    label_categories = None
    if "label_categories" in event:
        label_categories = event["labelCategories"]

    payload = event["payload"]
    role_arn = event["roleArn"]

    output_config = None  # Output s3 location. You can choose to write your annotation to this location
    if "outputConfig" in event:
        output_config = event["outputConfig"]

    # If you specified a KMS key in your labeling job, you can use the key to write
    # consolidated_output to s3 location specified in outputConfig.
    kms_key_id = None
    if "kmsKeyId" in event:
        kms_key_id = event["kmsKeyId"]

    # Create s3 client object
    s3_client = S3Client(role_arn, kms_key_id)
    
    print(payload)

    # Perform consolidation
    return do_consolidation(labeling_job_arn, payload, label_attribute_name, s3_client)
    
def do_consolidation(labeling_job_arn, payload, label_attribute_name, s3_client):
    """
        Core Logic for consolidation
    :param labeling_job_arn: labeling job ARN
    :param payload:  payload data for consolidation
    :param label_attribute_name: identifier for labels in output JSON
    :param s3_client: S3 helper class
    :return: output JSON string
    """
    # Extract payload data
    if "s3Uri" in payload:
        s3_ref = payload["s3Uri"]
        payload = json.loads(s3_client.get_object_from_s3(s3_ref))
    
    
    print(payload)
    
    # Payload data contains a list of data objects.
    # Iterate over it to consolidate annotations for individual data object.
    consolidated_output = []
    success_count = 0  # Number of data objects that were successfully consolidated
    failure_count = 0  # Number of data objects that failed in consolidation

    for p in range(len(payload)):
        try:
            # Notice that, no consolidation is performed, worker responses are combined and appended to final output
            # You can put your consolidation logic here
            
            dataset_object_id = payload[p]['datasetObjectId']
            input_seq_s3_uri = payload[p]['dataObject']['s3Uri']
            log_prefix = "[{}] data object id [{}] :".format(labeling_job_arn, dataset_object_id)

            annotations = payload[p]['annotations']
            
            # Iterate over annotations. Log all annotation to your CloudWatch logs
            for i in range(len(annotations)):
                worker_id = annotations[i]["workerId"]
                
                s3.put_object(
                     Body=json.dumps(annotations[i]),
                     Bucket=...,
                     Key=sg_outdir+"/"+input_seq_s3_uri.split("/")[-1].split("-")[0]+"/"+worker_id+'_'+curr_time+'.json'
                )
                
                annotation_content = annotations[i]['annotationData'].get('content')
                annotation_s3_uri = annotations[i]['annotationData'].get('s3uri')
                annotation = annotation_content if annotation_s3_uri is None else s3_client.get_object_from_s3(annotation_s3_uri)
                annotation_from_single_worker = json.loads(annotation)

                print("{} Received Annotations from worker [{}] is [{}]".format(log_prefix, worker_id, annotation_from_single_worker))


            consolidated_annotation = {"annotationsFromAllWorkers": annotations}
            
            
            # Build consolidation response object for an individual data object
            response = {
                "datasetObjectId": dataset_object_id,
                "consolidatedAnnotation": {
                    "content": {
                        label_attribute_name: consolidated_annotation
                    }
                }
            }

            success_count += 1
            #print("{} Consolidating annotations END ".format(log_prefix))

            # Append individual data object response to the list of responses.
            if response is not None:
                consolidated_output.append(response)

        except:
            failure_count += 1
            print(" Consolidation failed for dataobject {}".format(p))
            #print(" Unexpected error: Consolidation failed." + str(sys.exc_info()[0]))

    print("Consolidation Complete. Success Count {}  Failure Count {}".format(success_count, failure_count))
    print(" ------------------------- ")
    return consolidated_output