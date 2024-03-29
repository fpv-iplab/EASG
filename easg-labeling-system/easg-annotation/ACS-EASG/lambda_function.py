import json
import sys
from s3_helper import S3Client
import boto3
from datetime import datetime

sg_outdir = "data/not_validated_scene_graphs"
old_framedir = "data/frames"
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
    input_seq_s3_uri = ""
    # Payload data contains a list of data objects.
    # Iterate over it to consolidate annotations for individual data object.
    consolidated_output = []
    success_count = 0  # Number of data objects that were successfully consolidated
    failure_count = 0  # Number of data objects that failed in consolidation


    for p in range(len(payload)):
        response = None
        try:
            dataset_object_id = payload[p]['datasetObjectId']
            input_seq_s3_uri = payload[p]['dataObject']['s3Uri']
            log_prefix = "[{}] data object id [{}] :".format(labeling_job_arn, dataset_object_id)
            print("{} Consolidating annotations BEGIN ".format(log_prefix))
    
            annotations = payload[p]['annotations']
            for idx, annotation in enumerate(annotations):
                annotations[idx]['input_seq_s3_uri'] = input_seq_s3_uri
                s3.put_object(
                     Body=json.dumps(annotation),
                     Bucket=...,
                     Key=sg_outdir+"/"+input_seq_s3_uri.split("/")[-1].split("-")[0]+"/"+annotation['workerId']+'_'+curr_time+'.json'
                )
                
            print("{} Received Annotations from all workers {}".format(log_prefix, annotations))
    
            # Iterate over annotations. Log all annotation to your CloudWatch logs
            for i in range(len(annotations)):
                worker_id = annotations[i]["workerId"]
                annotation_content = annotations[i]['annotationData'].get('content')
                annotation_s3_uri = annotations[i]['annotationData'].get('s3uri')
                annotation = annotation_content if annotation_s3_uri is None else s3_client.get_object_from_s3(annotation_s3_uri)
                annotation_from_single_worker = json.loads(annotation)
    
    
            consolidated_annotation = {"annotationsFromAllWorkers": transform(annotations)[0]}
            
            
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
    
            # Append individual data object response to the list of responses.
            if response is not None:
                consolidated_output.append(response)

        except:
            failure_count += 1
            print(" Consolidation failed for dataobject {}".format(p))
            print(" Unexpected error: Consolidation failed." + str(sys.exc_info()[0]))

    print("Consolidation Complete. Success Count {}  Failure Count {}".format(success_count, failure_count))

    print(" -- Consolidated Output -- ")
    print(consolidated_output)
    print(" ------------------------- ")
    return consolidated_output
    
    
def transform(annotations):
    allAnnotations = []
    for _data in annotations:
        worker_id = _data['workerId']
        annotation_s3_ref = _data['input_seq_s3_uri']
        
        data = json.loads(_data['annotationData']['content'])
        subj = data['subj']
        dobj = data['dobj']
        verb = data['verb']
        clip_uid = data['clip_uid']
        ts = data['timestamp']
        
        pnr_frame_num = data['pnrFrameNumber']
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        newverb = data['newverb']
        newnoun = data['newnoun']
        
        sg = {'relations' : [], 'groundings' : {}, 'timestamp' : ts, 'clip_uid' : clip_uid, 'pnr_frame_num':pnr_frame_num,
        'imageHeight':image_height, 'imageWidth':image_width, 'newverb':newverb, 'newnoun':newnoun}
        
        relations = []
        groundings = {}
        pre_groundings, pnr_groundings, post_groundings = [],[],[]
        relations.append({'subj': {'object_name':subj,'instance_num':0}, 'predicate':verb, 'obj' : {'object_name':dobj,'instance_num':0}})
        for entry in json.loads(data['annotations']):
            try:
                object_name = entry['frames'][0]['bbs']['object'].split(":")[0]
                instance_num = int(entry['frames'][0]['bbs']['object'].split(":")[1])
            except:
                object_name = 'both_hands'
                instance_num = 0 
            relations.append({'subj': {'object_name':dobj, 'instance_num:':0}, 'predicate':entry['role'], 'obj' : {'object_name':object_name, 'instance_num':instance_num}})
            for frame in entry['frames']:
                obj_name = frame['bbs']['object'].split(":")[0]
                try:
                    inst_num = int(frame['bbs']['object'].split(":")[1])
                except:
                    inst_num = 0 
                if frame['frameType'] == 'pre':
                    pre_groundings.append({
                            'object':{'object_name':obj_name,'instance_num':inst_num},
                            'left':frame['bbs']['left'],
                            'top':frame['bbs']['top'],
                            'width':frame['bbs']['width'],
                            'height':frame['bbs']['height']})
                elif frame['frameType'] == 'pnr':
                    pnr_groundings.append({
                            'object':{'object_name':obj_name,'instance_num':inst_num},
                            'left':frame['bbs']['left'],
                            'top':frame['bbs']['top'],
                            'width':frame['bbs']['width'],
                            'height':frame['bbs']['height']})
                else:
                    post_groundings.append({
                            'object':{'object_name':obj_name,'instance_num':inst_num},
                            'left':frame['bbs']['left'],
                            'top':frame['bbs']['top'],
                            'width':frame['bbs']['width'],
                            'height':frame['bbs']['height']})
                groundings['pre_frame'] = pre_groundings
                groundings['pnr_frame'] = pnr_groundings
                groundings['post_frame'] = post_groundings
        sg['relations'] = relations
        sg['groundings'] = groundings
        sg['clip_uid'] = clip_uid
        sg['workerId'] = worker_id
        sg['annotation_s3_ref'] = annotation_s3_ref

        allAnnotations.append(sg)
    return allAnnotations, clip_uid
    
    
def copy_files(bucket_name, old_prefix, new_prefix):
    
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    for obj in bucket.objects.filter(Prefix=old_prefix):
        if obj.key.endswith('.jpg'):
            old_source = { 'Bucket': bucket_name,
                           'Key': obj.key}
            # replace the prefix
            new_key = obj.key.replace(old_prefix, new_prefix, 1)
            new_obj = bucket.Object(new_key)
            new_obj.copy(old_source)