"""Setup for Council — AI Agent Orchestrator."""

from setuptools import setup, find_packages

from council import __version__

setup(
    name="council",
    version=__version__,
    description="AI Agent Orchestrator — 4-agent council with GUI + FV pipeline",
    author="om",
    packages=find_packages(),
    python_requires=">=3.10",
    py_modules=["council_cli"],
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "rank-bm25>=0.2.2",
        "json-repair>=0.8.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "council=council_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
)
