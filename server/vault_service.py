import os
import hvac


class VaultService:
    def __init__(self):
        self.aws_bucket_name = ""
        self.aws_acccess_key_id = ""
        self.aws_secret_access_key = ""
        self.aws_session_token = ""
        self.aws_default_region = ""
        self.aws_s3_endpoint = ""

        self.load_secrets()

    def load_secrets(self):
        vault_addr = os.environ.get("VAULT_ADDR")
        vault_token = os.environ.get("VAULT_TOKEN")
        vault_top_dir = os.environ.get("VAULT_TOP_DIR")
        vault_relative_path = os.environ.get("VAULT_RELATIVE_PATH")
        vault_mount = os.environ.get("VAULT_MOUNT")

        if not vault_addr or not vault_token or not vault_top_dir or not vault_relative_path or not vault_mount:
            print("Vault Addr, Token, Top Dir, Relative Path, or Mount not set in environment variables. Skipping Vault loading.")
            return

        vault_client = hvac.Client(url=vault_addr, token=vault_token)

        # Assuming secrets are stored under "secret/data/surimi"
        secret_path = f"{vault_top_dir}/{vault_relative_path}"
        secret = vault_client.secrets.kv.v2.read_secret_version(
            path=secret_path, mount_point=vault_mount)

        if secret['data']['data'].get('AWS_BUCKET_NAME'):
            self.aws_bucket_name = secret['data']['data']['AWS_BUCKET_NAME']
        if secret['data']['data'].get('AWS_ACCESS_KEY_ID'):
            self.aws_acccess_key_id = secret['data']['data']['AWS_ACCESS_KEY_ID']
        if secret['data']['data'].get('AWS_SECRET_ACCESS_KEY'):
            self.aws_secret_access_key = secret['data']['data']['AWS_SECRET_ACCESS_KEY']
        if secret['data']['data'].get('AWS_SESSION_TOKEN'):
            self.aws_session_token = secret['data']['data']['AWS_SESSION_TOKEN']
        if secret['data']['data'].get('AWS_BUCKET_REGION'):
            self.default_region = secret['data']['data']['AWS_DEFAULT_REGION']
        if secret['data']['data'].get('AWS_S3_ENDPOINT'):
            self.aws_s3_endpoint = secret['data']['data']['AWS_S3_ENDPOINT']

        print(f"Loaded secrets from Vault.")
