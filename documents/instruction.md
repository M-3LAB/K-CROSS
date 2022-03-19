
# How to update base code from fedmed-light
```bash
git remote add fedmed-light https://github.com/M-3LAB/fedmed-light.git
git fetch fedmed-light
git log fedmed-light/main
git cherry-pick <commitHash>
```

> if your github is empty, you can pull the base code from fedmed-light
```bash
git pull fedmed-light main --allow-unrelated-histories
```