import os
import hvac
import logging

LOGGER_NAME = "VaultService"

class VaultService:
    """
    The VaultService is responsible for retrieving S3 secrets from the Vault.

    The secrets are stored in the environment variables. This makes them
    accessible for the `S3_Storage`.
    """
    @staticmethod
    def LoadVaultSecretsInEnvironmentVariables():
        """
        Store the Vault secrets in the environment variables. Specifically, the
        secrets include:
        - AWS_ACCESS_ID
        - AWS_SECRET_ACCESS_KEY

        The other secrets are from the `.env` file or retrieved by other means,
        e.g. users manually configuring them in the shell.
        """
        logger = logging.getLogger(LOGGER_NAME)
        
        vault_addr = os.environ.get("VAULT_ADDR")
        vault_token = os.environ.get("VAULT_TOKEN")
        vault_top_dir = os.environ.get("VAULT_TOP_DIR")
        vault_relative_path = os.environ.get("VAULT_RELATIVE_PATH")
        vault_mount = os.environ.get("VAULT_MOUNT")
        
        # Print useful error messages for troubleshooting
        # Verify that the variables individually.
        if not vault_addr:
            logger.error("Environment variable 'VAULT_ADDR' is not set.")
            return
        elif len(vault_addr.strip()) <= 0:
            # Empty env var means it is unset by os.environ.get(). We
            # therefore only check for white-space characters.
            logger.error("Environment variable 'VAULT_ADDR' has white-space characters only.")
            return
        
        if not vault_token:
            logger.error("Environment variable 'VAULT_TOKEN' is not set.")
            return
        elif len(vault_token.strip()) <= 0:
            logger.error("Environment variable 'VAULT_TOKEN' has white-space characters only.")
            return

        if not vault_top_dir:
            logger.error("Environment variable 'VAULT_TOP_DIR' is not set.")
            return
        elif len(vault_top_dir.strip()) <= 0:
            logger.error("Environment variable 'VAULT_TOP_DIR' has white-space characters only.")
            return

        if not vault_relative_path:
            logger.error("Environment variable 'VAULT_RELATIVE_PATH' is not set.")
            return
        elif len(vault_relative_path.strip()) <= 0:
            logger.error("Environment variable 'VAULT_RELATIVE_PATH' has white-space characters only.")
            return

        if not vault_mount:
            logger.error("Environment variable 'VAULT_MOUNT' is not set.")
            return
        elif len(vault_mount.strip()) <= 0:
            logger.error("Environment variable 'VAULT_MOUNT' has white-space characters only.")
            return
        
        vault_client = hvac.Client(url=vault_addr, token=vault_token)
        
        # Assuming secrets are stored under "secret/data/surimi"
        secret_path = f"{vault_top_dir}/{vault_relative_path}"
        secret = vault_client.secrets.kv.v2.read_secret_version(path=secret_path, mount_point=vault_mount)
        
        for key, value in secret['data']['data'].items():
            os.environ[key] = str(value)
            print(f"Loaded secret '{key}' from Vault into environment variables.")
