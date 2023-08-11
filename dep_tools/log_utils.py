def get_log_path(prefix: str, dataset_id: str, version: str, datetime: str) -> str:
    return f"{prefix}/{dataset_id}/logs/{dataset_id}_{version}_{datetime.replace('/', '_')}_log.csv"
