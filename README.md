# Project Title

Classification of Brain MR Images in parkinson desease

## Getting Started

This project consists of two phases:
1- skull stripping in MR images to extract brain
2- classification of parkinson deseas

### Dataset

PPMI dataset is used in this project:  

```
https://www.ppmi-info.org/access-data-specimens/download-data/
```

## Running the tests

First of all, all images have to be skull stripped. To do this, you have execute "skullstrip.py". All of the result of this part are saved in .npy format.

Finally for classification, you have to execute "mri.py". This piece of code will load the result of previous section and process them.


![alt text](https://github.com/alinouri91/Parkinson-deep-learning/blob/master/im1.tif)
### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

