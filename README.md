# iic

https://www.kaggle.com/puneet6060/intel-image-classification

## dev env

### export env

```
$ conda env export -n iic | grep -v "^prefix: " > ./environment.yml
```

### update env

```
$ conda env update -n iic
```

### create env

```
$ conda env create --name iic --file ./environment.yml
```

