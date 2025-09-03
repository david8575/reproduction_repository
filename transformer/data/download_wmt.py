import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path

def download_wmt_data():
    """
    WMT 2014 English-German 데이터 다운로드
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Europarl 데이터 (논문에서 사용)
    europarl_url = "http://www.statmt.org/wmt14/training-parallel-europarl-v7.tgz"
    europarl_path = data_dir / "europarl.tgz"
    
    if not europarl_path.exists():
        print("Europarl 데이터 다운로드 중...")
        urllib.request.urlretrieve(europarl_url, europarl_path)
        
        print("압축 해제 중...")
        with tarfile.open(europarl_path, "r:gz") as tar:
            tar.extractall(data_dir)
    
    print("데이터 다운로드 완료!")

if __name__ == "__main__":
    download_wmt_data()
