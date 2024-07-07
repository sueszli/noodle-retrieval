```
  _   _                 _ _        ____      _        _                 _ 
 | \ | | ___   ___   __| | | ___  |  _ \ ___| |_ _ __(_) _____   ____ _| |
 |  \| |/ _ \ / _ \ / _` | |/ _ \ | |_) / _ \ __| '__| |/ _ \ \ / / _` | |
 | |\  | (_) | (_) | (_| | |  __/ |  _ <  __/ |_| |  | |  __/\ V / (_| | |
 |_| \_|\___/ \___/ \__,_|_|\___| |_| \_\___|\__|_|  |_|\___| \_/ \__,_|_|
```

neural information retrieval and ranking & extractive question using the following models:

-   K-NRM: End-to-End Neural Ad-hoc Ranking with Kernel Pooling (Xiong et al., 2017: https://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf)
-   TK: Transformer-Kernel Ranking (Mitra et al., 2020: https://arxiv.org/pdf/2104.09393.pdf)

    based on: https://github.com/sebastian-hofstaetter/matchmaker

---

run `make` to see the available target environments to run the code.

```bash
make help
```

which target to choose?

-   conda:

    works great. no issues.

-   docker:

    works great, but using a gpu is more complicated.
    
    you won't have access to your vscode plugins as of may 2024.

    see: https://github.com/microsoft/vscode/issues/174632

-   google colab:

    not an option.

    see: https://github.com/sebastian-hofstaetter/teaching/discussions/129 (this might be a solution, but i was not able to get it to work)

    see: https://github.com/googlecolab/colabtools/issues/4212#issuecomment-1856302948

    see: https://www.reddit.com/r/GoogleColab/comments/13605i3/comment/jinqaz3/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
