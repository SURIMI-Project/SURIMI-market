import os
import hvac


class VaultService:
    def __init__(self):
        # TODO use vars from s3 storage.
        self.aws = ""

    def LoadVaultSecretsInEnvironmentVariables(self):
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
        secret = vault_client.secrets.kv.v2.read_secret_version(path=secret_path, mount_point=vault_mount)
        
        # TODO investigate whether this adheres to security guidelines, as it
        #      is easier for malicious actors to sniff environment variables
        #      than finding reading secrets from memory. The OS already
        #      implements a list of mechanism to defend against such attacks.
        for key, value in secret['data']['data'].items():
            os.environ[key] = str(value)
            print(f"Loaded secret '{key}' from Vault into environment variables.")

