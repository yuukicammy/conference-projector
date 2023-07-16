# Conference Projector

<a herf="https://modal.com/apps/yuukicammy/conference-projector/en">
<img src="https://github.com/yuukicammy/conference-projector/raw/main/data/readme/first_screen_smpl.png" height=150 alt="Top Screen" title="Top Screen"><br>
<b>Conference Projector</b></a>


## What is Conference Projector?

* Conference Projector is a web application that visualizes the all papers in the International Conference.
* You can search for papers that are closely related in terms of category and application based on the graph.
* Instead of text-based search, you can explore papers with a wide perspective.
* It also helps to grasp trending or niche areas within the conference.

## Features
* Display of scatter plots visualizing the all papers.
* Search for papers based on category or application.
* Display detailed information about papers and related papers.
* Graph updates based on user interaction.

## Demo

Currently only CVPR2023 is supported.

<a href="https://www.youtube.com/watch?v=k__408VdaDk">
<img src="https://github.com/yuukicammy/conference-projector/raw/main/data/readme/youtube_screen.png" height=150 alt="YouTube Screen" title="YouTube Screen"></a>

You can see a live demo of the website at the following link: [Conference Projector](https://yuukicammy--paper-viz-webapp-wrapper.modal.run)

## Tech Stack

Conference Projector is built using the following tools:

- Language: Python
- LLM: [OpenAI API](https://openai.com/blog/openai-api)
- Web App Framework: [Dash (Plotly)](https://dash.plotly.com/)
- DB: [Azure Cosmos](https://azure.microsoft.com/en-us/products/cosmos-db/)
- Infrastructure: [Modal](https://modal.com/)

## Usage

1. Sign up for an account at [modal.com](https://modal.com/) and follow the setup instructions to install the modal package and set your API key.

1. Sign up [Microsoft Azure ](https://azure.microsoft.com/en-us/free/cosmos-db/).

1. Sign up [OpenAI API](https://platform.openai.com/).

1. Clone this repository.
```
$ git clone https://github.com/your-username/conference-projector.git
$ cd conference-projector
```
5. Install the required packages in the local environment. (Note: Basically, the necessary libraries are automatically installed in the Modal container. The libraries installed here are mainly needed to define type hints.)
```
$ pip install -r requirements.txt
```
6. Prepare all data to be used for the website.
```
$ modal run src/pipeline.py
```
7. Deploy the website
```
$ modal deploy src/webapp.py
```

## License
This project is licensed under the MIT License. For more information, see the [LICENSE file](.MIT_License.txt).

#### CVPR Open Access 
> This material is presented to ensure timely dissemination of scholarly and technical work. Copyright and all rights therein are retained by authors or by other copyright holders. All persons copying this information are expected to adhere to the terms and constraints invoked by each author's copyright.

See [the official website](https://openaccess.thecvf.com/menu).

#### arXiv
Thank you to arXiv for use of its open access interoperability.


# ToDo
- [ ] Type hints with "" 
- [ ] English/Japanese settings
- [ ] Modify the prompt
- [ ] CI 
   - [x] lint 
   - [x] unittest
   - [ ] mypy
- [ ] CD
- [ ] Improve representative image extraction algorithm
- [ ] Mobile adaptation
- [x] Marker changes for important papers