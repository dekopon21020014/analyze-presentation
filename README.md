# analyze-presentation
```
git clone git@github.com:dekopon21020014/analyze-presentation.git
cp src/.env.sample src/.env # and fill the api key
cd analyze-presentation
docker build -t image-name:tag .
docker run --rm -it -v ./src:/app -p 8000 8000 image-name:tag 
```
