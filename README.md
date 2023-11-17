# PocariSweatのリモートリポジトリ

## Poetryの使用方法
### インストール
1. poetryを[こことかから](https://cocoatomo.github.io/poetry-ja/)インストールする
2. パスを通す ← **poetry パス windows**とかで検索すれば出てくると思います
3. ```poetry --version```でバージョンが表示されればok

### 仮想環境の作成とライブラリのインストール
1. ```git clone```とか```git pull```とかでリモートの最新の```pyproject.toml```と```poetry.lock```をローカルに持ってくる
2. ```poetry install```で仮想環境の作成とライブラリとかのインストールができる
3. ```poetry env list```とかで仮想環境が表示されればok

### 仮想環境への出入り
仮想環境へ入る
```bash
poetry shell
```
仮想環境から出る
```bash
exit
```

<!--
## pipとvenvを使って、仮想環境を作成し、仮想環境内でパッケージをインストールする

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

!-->
