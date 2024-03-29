from botocore.exceptions import ClientError
import boto3


class S3Client(object):
    """
     Helper Class for S3 operations
    """
    s3_client = boto3.client("s3")
    s3 = boto3.resource("s3")

    def __init__(self, role_arn=None, kms_key_id=None):
        """
        Initialize the S3 resource using provided Role and Kms Key
        :param role_arn: Role which have access to consolidation request S3 payload file.
        :param kms_key_id: KMS key if S3 bucket is encrypted
        :return:
        """
        DEFAULT_SESSION = "Custom_Annotation_Consolidation_Lambda_Session"
        sts_connection = boto3.client('sts')
        assume_role_object = sts_connection.assume_role(RoleArn=role_arn, RoleSessionName=DEFAULT_SESSION)
        session = boto3.Session(
            aws_access_key_id=assume_role_object['Credentials']['AccessKeyId'],
            aws_secret_access_key=assume_role_object['Credentials']['SecretAccessKey'],
            aws_session_token=assume_role_object['Credentials']['SessionToken'])
        self.s3 = session.resource('s3')
        self.s3_client = session.client('s3')
        self.kms_key_id = kms_key_id

    def put_object_to_s3(self, data, bucket, key, content_type):
        """
        Helper function to persist data in S3
        """
        try:
            if not content_type:
                # Default content type
                content_type = "application/octet-stream"
            image_object = self.s3.Object(bucket, key)
            if self.kms_key_id:
                image_object.put(Body=data, ContentType=content_type, SSEKMSKeyId=self.kms_key_id,
                                 ServerSideEncryption="aws:kms")
            else:
                image_object.put(Body=data, ContentType=content_type)
        except ClientError as e:
            raise ValueError("Failed to put data in bucket: {}  with key {}.".format(bucket, key), e)
        return "s3://" + image_object.bucket_name + "/" + image_object.key

    def get_object_from_s3(self, s3_url):
        """ Helper function to retrieve data from S3 """
        bucket, path = S3Client.bucket_key_from_s3_uri(s3_url)

        try:
            payload = self.s3_client.get_object(Bucket=bucket, Key=path).get('Body').read().decode('utf-8')
        except ClientError as e:
            print(e)
            if e.response['Error']['Code'] == "404" or e.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                raise ValueError("Failed to retrieve data from {}.".format(s3_url), e)

        return payload

    @staticmethod
    def bucket_key_from_s3_uri(s3_path):
        """ Return bucket and key from s3 URL
        Parameters
        ----------
        s3_path: str, required
            s3 URL of data object ( image/video/text/audio etc )
        Returns
        ------
        bucket: str
            S3 Bucket of the passed URL
        key: str
            S3 Key of the passed URL
        """
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)

        return bucket, key