# PocariSweatのリモートリポジトリ

## ~~pipとvenvを使って、仮想環境を作成し、仮想環境内でパッケージをインストールする~~
venvは仮想環境が作れるやつ(インストールは多分いらない)
```bash
python -m venv --help
```
を実行すると、helpページが出力されるはず(されたらok)

### 1.仮想環境を作成する。
プロジェクトのディレクトリ(ポカリスウェット)へ行って以下を実行
``` bash
py -m venv env
```

### 2.仮想環境に入る
``` bash
.\env\Scripts\activate
```

### 3.仮想環境にパッケージをインストール(requirement.txtに記述してあるやつがインストールされる)
``` bash
pip install -r requirements.txt
```
実行することで、パッケージを一括インストールできる。

### 補足
- 仮想環境に入る
``` bash
.\env\Scripts\activate
```
- 仮想環境から出る
``` bash
deactivate
```
