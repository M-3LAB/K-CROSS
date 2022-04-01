
# How to update base code from fedmed-light
```bash
git remote add fedmed-light https://github.com/M-3LAB/fedmed-light.git
git fetch fedmed-light
git log fedmed-light/main
git cherry-pick <commitHash>
```

<<<<<<< HEAD
> if your github is empty, you can pull the base code from fedmed-light
=======
> you can also pull the base code from fedmed-light, all changes
>>>>>>> cbbd2f4f26a351aed1b678ac13ec43a0f7c23b65
```bash
git pull fedmed-light main --allow-unrelated-histories
```