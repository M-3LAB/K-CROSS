
# How to update base code from fedmed-light
```bash
git remote add fedmed-light https://github.com/M-3LAB/fedmed-light.git
git fetch fedmed-light
git log fedmed-light/main
git cherry-pick <commitHash>
```

> you can also pull the base code from fedmed-light, all changes
```bash
git pull fedmed-light main --allow-unrelated-histories
```