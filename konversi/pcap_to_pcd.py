from helper import *
from pathlib import Path

for folder in Path('20230301/45Deg/Pose').iterdir():
    tmp = Path(folder).stem
    # print(folder, tmp)

    for subfolder in Path(folder).rglob('*.json'):
        meta_file = Path(subfolder).stem
        f = os.path.join(folder, meta_file)
        meta_path = f+".json"
        pcap_path = f+".pcap"
        with open(meta_path, 'r') as f:
            info = client.SensorInfo(f.read())

        source = pcap.Pcap(pcap_path, info)
        meta = source.metadata
        scans = client.Scans(source)
        if not os.path.exists("20230301/PCD/"):
            os.makedirs("20230301/PCD/")

        if not os.path.exists("20230301/PCD/45Deg/"):
            os.makedirs("20230301/PCD/45Deg/")

        if not os.path.exists("20230301/PCD/45Deg/"+tmp):
            os.makedirs("20230301/PCD/45Deg/"+tmp)

        write_path = "20230301/PCD/45Deg/"+str(tmp)
        print("Writing PCD to: ",write_path)
        pcap_to_pcd(source=source,metadata=meta, pcd_base=write_path+"/obj")