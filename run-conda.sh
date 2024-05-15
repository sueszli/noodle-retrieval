# ------------------------------------------- decompress data
# validate ./data/* files
if [ ! -d data ]; then echo "data/ directory not found"; exit 1; fi
if ! ls data/*-chunk-* &> /dev/null && ! ls data/*.md5 &> /dev/null; then echo "invalid files found in data/"; exit 1; fi

# create data-merged directory
rm -rf data-merged
mkdir data-merged
echo "created data-merged directory"

# merge chunks into data-merged directory
cat data/*-chunk-* > data-merged/merged.tar.gz
echo "merged chunks into data-merged/merged.tar.gz"

# validate checksum
expected_checksum=$(cat data/*.md5)
actual_checksum=$(md5sum data-merged/merged.tar.gz | awk '{ print $1 }')
if [ $expected_checksum != $actual_checksum ]; then echo "checksum mismatch"; exit 1; fi
echo "checksum matched: $expected_checksum == $actual_checksum"

# untar in data-merged
tar -xzf data-merged/merged.tar.gz -C data-merged
rm data-merged/merged.tar.gz
echo "untarred data-merged/merged.tar.gz"

# ------------------------------------------- install conda
brew install --cask miniconda
conda update conda

conda init zsh
conda init bash
exit # restart shell

conda config --set auto_activate_base false # disable auto-activation
conda config --env --set subdir osx-64 # emulate x86_64

# ------------------------------------------- start
conda activate base

conda create --yes --name noodle-retrieval python=3.6.12 anaconda
conda activate noodle-retrieval

pip install -r requirements.txt

# ------------------------------------------- stop
conda deactivate
conda remove --yes --name noodle-retrieval --all

# ------------------------------------------- verify cleanup
conda env list
