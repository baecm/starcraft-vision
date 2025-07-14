import paramiko

def fetch_gpu_memory_usage(client):
    cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
    stdin, stdout, stderr = client.exec_command(cmd)
    error = stderr.read().decode()
    if error:
        raise RuntimeError(error)

    usage = []
    for line in stdout.read().decode().strip().splitlines():
        if not line.strip():
            continue
        used, total = map(int, line.strip().split(","))
        usage.append((used, total))
    return usage

def fetch_disk_usage(client, path="/"):
    cmd = f"df -h {path}"
    stdin, stdout, stderr = client.exec_command(cmd)
    error = stderr.read().decode()
    if error:
        raise RuntimeError(error)
    return stdout.read().decode()

def print_gpu_usage_bar(idx, used, total, bar_width=30):
    ratio = used / total
    fill = int(ratio * bar_width)
    bar = "#" * fill + "-" * (bar_width - fill)
    print(f"    GPU {idx:<2}: {used:>6} MiB / {total:>6} MiB ({ratio:>6.2%}) |{bar}|")

def print_disk_usage(output):
    print("  Disk usage:")
    for line in output.strip().splitlines():
        print("   ", line)

def main():
    host_map = {
        f"172.27.186.{i}": f"worker{(i - 220):02d}" for i in range(220, 230)
    }
    host_map["172.27.186.230"] = "DGX-Station"

    for host, name in host_map.items():
        print(f"[{name} ({host})]")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username="baecm", password="cilab", timeout=5)

            # 1) GPU 사용량 조회
            gpu_usages = fetch_gpu_memory_usage(client)
            for idx, (used, total) in enumerate(gpu_usages):
                print_gpu_usage_bar(idx, used, total)

            # 2) 디스크 사용량 조회 (루트 파티션)
            disk_output = fetch_disk_usage(client, path="/")
            print_disk_usage(disk_output)

        except Exception as e:
            print(f"  Error: {e}")
        finally:
            client.close()
        print()  # 호스트 구분 여백

if __name__ == "__main__":
    main()
