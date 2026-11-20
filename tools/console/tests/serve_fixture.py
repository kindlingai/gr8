"""Run the console against synthetic HTTP + TCP fixtures for browser tests."""
import argparse
import uvicorn
from tools.console.server.main import create_app
from tools.console.server.service import Console
from tools.console.tests.fakes import Fixture

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    fixture = Fixture()
    console = Console(fixture.console_config, fixture)
    try:
        uvicorn.run(create_app(console=console), host="localhost", port=args.port)
    finally:
        fixture.close()
