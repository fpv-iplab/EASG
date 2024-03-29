import json
import boto3
import os

s3 = boto3.resource('s3')
sg_dir = "data/not_validated_scene_graphs"

def lambda_handler(event, context):
    print("*********************** Called PRE-EASG-VALIDATION lambda function ***********************")
    bucket_name, object_key = get_bucket_object_key(event["dataObject"]["source-ref"])
    
    response = {
        'taskInput': {
            "taskObject": event["dataObject"]["source-ref"],
            "video_uid" : "",
            "fps": "",
            "srcImages" : {
                "preFrame" : {
                    "s3Uri" : ""
                },
                "pnrFrame" : {
                    "s3Uri" : ""
                },
                "postFrame" : {
                    "s3Uri" : ""
                }
            },
            "sgAnnotations": [],
            "pnrFrameNumber" : "",
            "clipS3Uri" : "",
            "clipUid" : "",
            "questions":[],
            "contradictions":[]
        },
        'isHumanAnnotationRequired': 'true'} 

    jsonObj = read_json(bucket_name,object_key)
    
    frames = jsonObj['frames']
    prefix = jsonObj['prefix']
    
    clip_s3_uri, annotations_s3_uri = "",""
    annotation_uid = event["dataObject"]["source-ref"].split("/")[-1].split("-")[0]
    
    print(f"Annotation UID: {annotation_uid}")
    video_uid=""
    if len(frames)>0:
        clip_uid = frames[0]['frame'].split("_")[0]
        clip_s3_uri = get_clip_s3_uri(bucket_name,clip_uid)
        
        _, mapping_clip_video_object_key = get_bucket_object_key("s3://{bucket_name}/data/clip_video_uids_mapping.json")
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
        
    response['taskInput']['clipS3Uri'] = clip_s3_uri
    response['taskInput']['clipUid'] = clip_s3_uri.split("/")[-1].split("_")[0].rstrip(".mp4")
    
    response['taskInput']['srcImages']['preFrame']['s3Uri'] = pre_frame_s3_uri#.replace("frames","not_validated_frames")
    response['taskInput']['srcImages']['pnrFrame']['s3Uri'] = pnr_frame_s3_uri#.replace("frames","not_validated_frames")
    response['taskInput']['srcImages']['postFrame']['s3Uri'] = post_frame_s3_uri#.replace("frames","not_validated_frames")
    
    response['taskInput']['video_uid'] = video_uid
    response['taskInput']['fps'] = 30
    
    response['taskInput']['questions'] = []
    merged_graph, verb, noun, pnrFrameNumber = jsons_to_str(bucket_name, sg_dir, annotation_uid)
    response['taskInput']['pnrFrameNumber'] = pnrFrameNumber
    # print('MERGED GRAPH: ',merged_graph)

    questions, contradictions = generate_questionnaire(merged_graph, verb, noun)
    
    response['taskInput']['questions'] = questions
    response['taskInput']['contradictions'] = contradictions
    
    return response
        

def get_clip_s3_uri(bucket_name, clip_uid):
    return f's3://{bucket_name}/data/clips/{clip_uid}.mp4'

def get_annotations_s3_uri(bucket_name, annotation_uid, clip_uid):
    return f's3://{bucket_name}/data/annotations/{annotation_uid}/{clip_uid}.json'

    
def get_critical_frame_s3_uris(bucket_name, annotation_uid, frame_filenames):
    response = {'pre':'','pnr':'','post':''}
    for f in frame_filenames:
        if f['frame-no'] == 0: response['pnr'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
        elif f['frame-no'] == 1: response['post'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
        elif f['frame-no'] == 2: response['pre'] = os.path.join("s3://"+bucket_name+"/data/frames/"+annotation_uid,f['frame'])
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

def jsons_to_str(bucket_name, sg_dir, annotation_uid): # bucket_name, sg_dir, annotation_uid
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    prefix = sg_dir+"/"+annotation_uid
    sgs = []
    verb = ""
    noun = ""
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith('.json'):
            data = read_json(bucket_name,obj.key)
            annotations_string = data['annotationData']['content']
            annotations = json.loads(annotations_string)
            g, verb, noun, pnrFrameNumber = json_to_triplets(annotations)
            sgs += g
    if len(sgs) > 0:
        g_triplets = sort_phrases(list(set(sgs)))
        merged_graph = '; '.join(g_triplets)
    else:
        merged_graph = ""
    return merged_graph, verb, noun, pnrFrameNumber
            

def json_to_triplets(annotations):
    g = []
    print(annotations)
    verb = annotations['verb'].replace('-',' ')
    noun = annotations['dobj'].replace('-',' ')
    pnrFrameNumber = annotations['pnrFrameNumber']
    try:
        newverb = annotations['newverb']
        newnoun = annotations['newnoun']
        if newverb!='none' or newnoun!='none':
            g.append('Camera wearer - verb - {}'.format(newverb.replace('-',' ')))
            g.append('{} - direct object - {}'.format(newverb.replace('-',' '), newnoun))
            return g, verb, noun, pnrFrameNumber
    except:
        return g, None, None, pnrFrameNumber
    noun = annotations['dobj']
    roles_objects = json.loads(annotations['annotations'])
    if len(roles_objects):
        g.append('Camera wearer - verb - {}'.format(verb.replace('-',' ')))
        g.append('{} - direct object - {}'.format(verb.replace('-',' '), noun))
        for role_obj in roles_objects:
            if len(role_obj['frames']):
                indir_obj_all = role_obj['frames'][0]['bbs']
                indir_obj = indir_obj_all['object']
                if indir_obj.find('hand')==-1:
                    g.append('{} - {} - {}'.format(verb.replace('-',' '), role_obj['role'], indir_obj.replace(':0','')))
                else:
                    g.append('{} - {} - {}'.format(verb.replace('-',' '), role_obj['role'], indir_obj.replace(':0','')))
    return g, verb, noun, pnrFrameNumber

    
def sort_phrases(phrases):
    # Define custom sort key function
    def sort_key(phrase):
        if 'verb' in phrase:
            return 1  # Priority 1 for "verb"
        elif 'direct object' in phrase:
            return 2  # Priority 2 for "direct object"
        else:
            return 3  # Lowest priority for other phrases
    # Sort the list using the custom key
    sorted_phrases = sorted(phrases, key=sort_key)
    
    return sorted_phrases


def convert_to_third_person(verb):
    """
    Converts a verb to its third person singular form according to basic English rules.
    This version also handles phrasal verbs.
    """
    # Split the verb phrase into parts
    parts = verb.split()

    # Apply the third person conversion to the first part (main verb)
    main_verb = parts[0]
    if main_verb[-1] == 'y' and main_verb[-2] not in 'aeiou':
        parts[0] = main_verb[:-1] + 'ies'
    elif main_verb[-1] in 'osxz' or main_verb[-2:] in ['ch', 'sh']:
        parts[0] = main_verb + 'es'
    else:
        parts[0] = main_verb + 's'
    
    # Reassemble the phrase if it was a composed verb
    if len(parts) > 1:
        return ' '.join(parts)
    else:
        return parts[0]



def prioritize_string(l, v, n):
    for i, string in enumerate(l):
        if v in string and n in string:
            # If both substrings are found, move the string to the first position
            l.insert(0, l.pop(i))
            break
    return l
    
def detect_contradictions(graph_str):
    graph_str = graph_str.replace('_',' ')
    triplets = [triplet.strip() for triplet in graph_str.split(';') if triplet.strip()]

    # Dictionary to hold verbs and direct objects for each subject
    subjects = {}

    # Tracking for other types of contradictions
    edges = {}
    hands = set()
    contradictions = []
    verbs = set()
    direct_objects = set()
    indirect_objects = set()
    for triplet in triplets:
        subj, rel, obj = triplet.split(' - ')
        # Collect verbs and their direct objects
        if rel in ['verb', 'direct object']:
            if rel == 'verb':
                verbs.add(obj)
            else:
                direct_objects.add(obj)
        else:
            if obj not in ['left hand', 'right hand', 'both hands']:
                indirect_objects.add(obj)
    # Detect Type 1 contradictions after collecting all verbs and objects
    if len(verbs) > 1 or len(direct_objects) > 1:
        options1 = []
        # Generate all combinations of verbs and direct objects for the subject
        for verb in verbs:
            for direct_object in direct_objects:
                options1.append(verb+" - "+direct_object)
        contradictions.append((1, options1))
    
    options2 = []
    options3 = []
    for triplet in triplets:
        subj, rel, obj = triplet.split(' - ')
        if (subj, obj) in edges and edges[(subj, obj)] != rel:
            if len(options2)==0:
                options2.append(' - '.join([subj,edges[(subj, obj)],obj]))
            options2.append(' - '.join([subj,rel,obj]))
        edges[(subj, obj)] = rel
        
        # Type 3: Contradiction with "with" edge not involving hands correctly
        if rel == 'with' and obj not in ['left hand', 'right hand', 'both hands']:
            options3.append(' - '.join([subj,rel,obj]))
            for direct_object in direct_objects:
                options3.append(' - '.join([direct_object,rel,obj]))
            for indirect_object in indirect_objects:
                if obj!=indirect_object:
                    options3.append(' - '.join([indirect_object,rel,obj]))
        
        # Type 4
        if rel == 'with' and obj in ['left hand', 'right hand', 'both hands']:
            hands.add(obj)
            
    if len(options2):
        contradictions.append((2, options2))
    if len(options3):
        contradictions.append((3, options3))
    if len(hands) > 1:
        contradictions.append((4, list(hands)))

    return contradictions
    
def generate_questionnaire(graph_str, verb, noun):
    contradictions = detect_contradictions(graph_str)
    questionnaire = {}
    q_idx = 0

    for contradiction in contradictions:
        if 1 in contradiction:
            # Generate question for Type 1 contradiction
            question = "Select the correct statement:"
            options1 = prioritize_string(contradiction[1],verb,noun)
            answers = []
            for option in options1:
                v,n = option.split(' - ')
                answers.append('Camera wearer {} {}'.format(convert_to_third_person(v),n))
            
            questionnaire[q_idx] = {'type': 1, 'question': question, 'answers': answers}

        elif 2 in contradiction:
            # Generate question for Type 2 contradiction
            question = "Which of the statement is correct?"
            answers = []
            for ans in contradiction[1]:
                s,r,o = ans.split(' - ')
                answers.append('Camera wearer {} {} {} {}'.format(convert_to_third_person(verb),noun,r,o))
            questionnaire[q_idx] = {'type': 2, 'question': question, 'answers': answers}

        elif 3 in contradiction:
            question = "Select (grammatically) correct statement:"
            answers = []
            for ans in contradiction[1]:
                s,r,o = ans.split(' - ')
                if s==verb:
                    answers.append('Camera wearer {} {} {}'.format(convert_to_third_person(verb),r,o))
                else:
                    answers.append('{} is {} {}'.format(s,r,o))
            questionnaire[q_idx] = {'type': 3, 'question': question, 'answers': answers}
        
        elif 4 in contradiction:
            # Generate question for Type 4 contradiction
            question = "With which hand does Camera Wearer {} {}?".format(verb,noun)
            answers = contradiction[1]
            questionnaire[q_idx] = {'type': 4, 'question': question, 'answers': answers}

        q_idx += 1

    return questionnaire, contradictions