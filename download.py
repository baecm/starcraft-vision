import paramiko
import os
import stat
import time

REMOTE_DIR = "/mnt/nas/baecm/starcraft-vision/models"
LOCAL_DIR = "./models"
EXCLUDED_EXTENSIONS = [".pth"]

hostname = "172.27.186.230"
port = 22
username = "baecm"
password = "cilab"

def should_download(filename):
    return not any(filename.endswith(ext) for ext in EXCLUDED_EXTENSIONS)

def download_recursive(sftp, remote_path, local_path):
    os.makedirs(local_path, exist_ok=True)

    for entry in sftp.listdir_attr(remote_path):
        remote_item = f"{remote_path}/{entry.filename}"
        local_item = os.path.join(local_path, entry.filename)

        if stat.S_ISDIR(entry.st_mode):
            download_recursive(sftp, remote_item, local_item)
        else:
            if should_download(entry.filename):
                if os.path.exists(local_item):
                    local_size = os.path.getsize(local_item)
                    remote_size = entry.st_size

                    if local_size == remote_size:
                        print(f"Skipping existing identical file: {local_item}")
                        continue
                    else:
                        print(f"Overwriting changed file: {local_item}")

                else:
                    print(f"Downloading {remote_item} → {local_item}")

                sftp.get(remote_item, local_item)
                

def download_files():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)

    sftp = ssh.open_sftp()
    download_recursive(sftp, REMOTE_DIR, LOCAL_DIR)
    sftp.close()
    ssh.close()
    print("[Sync] Download complete.")

def monitor_and_download(interval=60):
    while True:
        print(f"[Monitor] Checking for new files in: {REMOTE_DIR}")
        try:
            download_files()
        except Exception as e:
            print(f"[Error] {e}")
        print(f"[Monitor] Waiting {interval} seconds...\n")
        time.sleep(interval)

if __name__ == "__main__":
    monitor_and_download(interval=60 * 60 * 2)
