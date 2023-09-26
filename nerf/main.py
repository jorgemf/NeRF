import typer
from .dataset import app as dataset_app
from .train import train
from .test import test
from .config import setup

# typer set up the command line interface
app = typer.Typer()
app.add_typer(dataset_app, name="dataset")
app.command()(train)
app.command()(test)


@app.command
def inference():
    pass


if __name__ == '__main__':
    setup()
    app()
