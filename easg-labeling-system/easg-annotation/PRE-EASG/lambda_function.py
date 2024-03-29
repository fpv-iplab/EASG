import json
import boto3
import os

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    print("*********************** Called PRE-EASG lambda function ***********************")
    bucket_name, object_key = get_bucket_object_key(event["dataObject"]["source-ref"])
    
    response = {
        'taskInput': {
            "taskObject": event["dataObject"]["source-ref"],
            "video_uid" : "",
            "fps": "",
            "allObjects" : "",
            "allTools": "",
            "srcImages" : {
                "preFrame" : {
                    "s3Uri" : "",
                    "clipFrameNo" : "",
                    "width" : "",
                    "height" : "",
                    "bbox" : []
                },
                "pnrFrame" : {
                    "s3Uri" : "",
                    "clipFrameNo" : "",
                    "width" : "",
                    "height" : "",
                    "bbox" : []
                },
                "postFrame" : {
                    "s3Uri" : "",
                    "clipFrameNo" : "",
                    "width" : "",
                    "height" : "",
                    "bbox" : []
                }
            },
            "clipS3Uri" : "",
            "clipUid" : "",
            "narrations_infos" : [],
        },
        'isHumanAnnotationRequired': 'true'} 
    print('Bucket name and Object key: ', bucket_name,object_key)   
    jsonObj = read_json(bucket_name,object_key)
    
    frames = jsonObj['frames']
    prefix = jsonObj['prefix']
    
    clip_s3_uri, annotations_s3_uri = "",""
    annotation_uid = event["dataObject"]["source-ref"].split("/")[-1].split("-")[0]

    print(f"Annotation UID: {annotation_uid}")
    
    if len(frames)>0:
        clip_uid = frames[0]['frame'].split("_")[0]
        clip_s3_uri = get_clip_s3_uri(bucket_name,clip_uid)
        
        _, mapping_clip_video_object_key = get_bucket_object_key("s3://{}/data/clip_video_uids_mapping.json".format(bucket_name))
        jsonMapping = read_json(bucket_name, mapping_clip_video_object_key)
        
        if jsonMapping:
            try:
                video_infos = jsonMapping[clip_uid]
                video_uid = video_infos[0]
                fps = video_infos[1]
            except:
                video_uid,fps = "",60
    
    res = get_critical_frame_s3_uris(bucket_name,annotation_uid,frames)
    pre_frame_s3_uri,pnr_frame_s3_uri,post_frame_s3_uri="","",""
    if res['pre']:
        pre_frame_s3_uri = res['pre']
    if res['pnr']:
        pnr_frame_s3_uri = res['pnr']
    if res['post']:
        post_frame_s3_uri = res['post']
        
    
    if annotation_uid and clip_uid:
        annotations_s3_uri = get_annotations_s3_uri(bucket_name,annotation_uid,clip_uid)
        
    _, annotations_object_key = get_bucket_object_key(annotations_s3_uri)
    jsonAnnotations = read_json(bucket_name, annotations_object_key)
    
    _, narrations_annotations_mappings_object_key = get_bucket_object_key("s3://{}/data/narration_annotation_mappings_unict_all.json".format(bucket_name))
    jsonNarrationsAnnotationsMappings = read_json(bucket_name, narrations_annotations_mappings_object_key)
    
    response['taskInput']['clipS3Uri'] = clip_s3_uri
    response['taskInput']['clipUid'] = clip_s3_uri.split("/")[-1].split("_")[0]
    response['taskInput']['srcImages']['preFrame']['s3Uri'] = pre_frame_s3_uri
    response['taskInput']['srcImages']['preFrame']['clipFrameNo'] = jsonAnnotations['pre_frame']['clip_frame_number']
    response['taskInput']['srcImages']['preFrame']['width'] = jsonAnnotations['pre_frame']['width']
    response['taskInput']['srcImages']['preFrame']['height'] = jsonAnnotations['pre_frame']['height']
    response['taskInput']['srcImages']['preFrame']['bbox'] = jsonAnnotations['pre_frame']['bbox']
    
    response['taskInput']['srcImages']['pnrFrame']['s3Uri'] = pnr_frame_s3_uri
    response['taskInput']['srcImages']['pnrFrame']['clipFrameNo'] = jsonAnnotations['pnr_frame']['clip_frame_number']
    response['taskInput']['srcImages']['pnrFrame']['width'] = jsonAnnotations['pnr_frame']['width']
    response['taskInput']['srcImages']['pnrFrame']['height'] = jsonAnnotations['pnr_frame']['height']
    response['taskInput']['srcImages']['pnrFrame']['bbox'] = jsonAnnotations['pnr_frame']['bbox']
    
    response['taskInput']['srcImages']['postFrame']['s3Uri'] = post_frame_s3_uri
    response['taskInput']['srcImages']['postFrame']['clipFrameNo'] = jsonAnnotations['post_frame']['clip_frame_number']
    response['taskInput']['srcImages']['postFrame']['width'] = jsonAnnotations['post_frame']['width']
    response['taskInput']['srcImages']['postFrame']['height'] = jsonAnnotations['post_frame']['height']
    response['taskInput']['srcImages']['postFrame']['bbox'] = jsonAnnotations['post_frame']['bbox']
    
    narrations_infos_full = jsonAnnotations['narrations_infos']
    narrations_infos_list = []
    correct_narration_text = ""
    narrations_annotations_mappings = jsonNarrationsAnnotationsMappings['mappings']
    for namapping in narrations_annotations_mappings:
        if list(namapping.keys())[0] == annotation_uid:
            narrations_infos = []
            narrations_infos.append(namapping[annotation_uid])
            correct_narration_text = namapping[annotation_uid]['narration_text']
    _, verb_preps_json_key = get_bucket_object_key("s3://{}/data/verb_preps.json".format(bucket_name)) 
    _, prep_questions_json_key = get_bucket_object_key("s3://{}/data/prep_questions.json".format(bucket_name))
    _, binary_prep_questions_json_key = get_bucket_object_key("s3://{}/data/binary_prep_questions.json".format(bucket_name))
    _, mappings_verb_prep_json_key = get_bucket_object_key("s3://{}/data/definitive_mappings_v3.json".format(bucket_name))
    verbPrepsObj = read_json(bucket_name, verb_preps_json_key)
    prepQuestObj = read_json(bucket_name, prep_questions_json_key)
    binaryPrepQuestObj = read_json(bucket_name, binary_prep_questions_json_key)
    mappingsVerbPrepObj = read_json(bucket_name, mappings_verb_prep_json_key)
    
    i = 0 
    allObjects = []
    for narr_info_full in narrations_infos_full:
        allObjects+=[list(o.keys())[0] for o in narr_info_full['preposition_object_pairs']]
    narration_texts = []
    for narr_info in narrations_infos:
        if narr_info['narration_text'] not in narration_texts:
            narration_texts.append(narr_info['narration_text'])
            
            if narr_info['verb'] in verbPrepsObj:
                preps = verbPrepsObj[narr_info['verb']]
                preps_questions_map = {}
                binary_questions_map = {}
                for p in preps:
                    try:
                        preps_questions_map[p] = replace_tokens(prepQuestObj[p],narr_info['verb'],narr_info['direct_object'])
                    except:
                        pass
                for bp in preps:
                    try:
                        if bp != 'with':
                            binary_questions_map[bp] = replace_tokens(binaryPrepQuestObj[bp],narr_info['verb'],narr_info['direct_object'])   
                    except:
                        pass
                narrations_infos_list.append({
                    'narration_id' : 'n_'+str(i),
                    'narration_text':narr_info['narration_text'],
                    'subject': narr_info['subject'],
                    'verb':narr_info['verb'],
                    'third_person_verb':make_3rd_form(narr_info['verb']),
                    'direct_object':narr_info['direct_object'],
                    'action_clip_start_sec':narr_info['action_clip_start_sec'],
                    'action_clip_end_sec':narr_info['action_clip_end_sec'],
                    'objects':[list(o.keys())[0] for o in narr_info['preposition_object_pairs']],
                    'preposition_object_pairs':narr_info['preposition_object_pairs'],
                    'questions':preps_questions_map,
                    'binary_questions':binary_questions_map,
                    'roles' : mappingsVerbPrepObj[narr_info['verb']]['preps'],
                    'mappings' : mappingsVerbPrepObj[narr_info['verb'].split("-")[0]]['mappings']
                })
                i += 1 
                

    filteredAllObject = []
    for o in allObjects:
        if o.strip() != "hand" and o.strip() != "hands":
            filteredAllObject.append(o.strip())
            
    allTools = filteredAllObject.copy()
    
    if 'right_hand' in allTools:
        allTools.remove('right_hand')
    if 'left_hand' in allTools:
        allTools.remove('left_hand')
    
            
    response['taskInput']['narrations_infos'] = narrations_infos_list
    response['taskInput']['video_uid'] = video_uid
    response['taskInput']['fps'] = 30
    response['taskInput']['allObjects'] = list(set(filteredAllObject))
    response['taskInput']['allTools'] = list(set(allTools))
    print("*********************** RESPONSE ***********************")
    print(response)

    return response
        
        
def get_bucket_object_key(source_ref):
    toks = [t for t in source_ref.split("/") if t]
    bucket_name = toks[1]
    object_key = '/'.join(toks[2:])
    return bucket_name, object_key
    
def read_json(bucket_name, object_key):
    obj = s3.Bucket(bucket_name).Object(object_key)
    jsonStr = obj.get()['Body'].read().decode('utf-8')
    jsonObj = json.loads(jsonStr)
    return jsonObj
    
def get_critical_frame_s3_uris(bucket_name, annotation_uid, frame_filenames):
    response = {'pre':'','pnr':'','post':''}
    for f in frame_filenames:
        if f['frame-no'] == 0: response['pnr'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
        elif f['frame-no'] == 1: response['post'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
        elif f['frame-no'] == 2: response['pre'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
    return response
        
        
def get_clip_s3_uri(bucket_name, clip_uid):
    return f's3://{bucket_name}/data/clips/{clip_uid}.mp4'

def get_annotations_s3_uri(bucket_name, annotation_uid, clip_uid):
    return f's3://{bucket_name}/data/annotations/{annotation_uid}/{clip_uid}.json'

def replace_tokens(question_text, verb, dobj):
    return question_text.replace('<VERB>',verb).replace('<DOBJ>',dobj)
    
def make_3rd_form(word):
    is_phrasal_verb = False
    participle = ""
    if len(word.split("-"))==2:
        participle = word.split("-")[1]
        is_phrasal_verb = True
        word = word.split("-")[0]
    if word[len(word)-1] == 'o' or word[len(word)-1] == 's' or word[len(word)-1] == 'x' or word[len(word)-1] == 'z':
        word = word + 'es'
    elif word[len(word)-2] == 'c' and word[len(word)-1] == 'h':
        word = word + 'es'
    elif word[len(word)-2] == 's' and word[len(word)-1] == 'h':
        word = word + 'es'
    elif word[len(word)-1] == 'y':
        word = word[:-1]
        word = word + 'ies'
    else:
        word = word + 's'
    if is_phrasal_verb and len(participle)>0:
        word = word + "-" + participle
    return word