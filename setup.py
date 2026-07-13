 # Makes the project installable as a local package
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Read dependencies directly from requirements.txt for single-source management
with open("requirements.txt", "r", encoding="utf-8") as f:
    required_packages = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="telecom_rag_bot",
    version="1.0.0",
    author="Enterprise Architecture Team",
    author_email="architecture@telecom-corp.internal",
    description="Production-grade agentic RAG chatbot utilizing LangGraph, Qdrant, PostgreSQL, and Gemini Flash.",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com",
    
    # Automatically locate code inside the src directory
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # Python runtime boundary definition
    python_requires=">=3.10",
    
    # Inject parsed application requirements
    install_requires=required_packages,
    
    # Categorized logical groupings for isolated environments
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.9.0",
            "httpx>=0.27.0"  # Crucial for async API endpoint testing
        ]
    },
    
    # Console scripts to fire system pipelines from a terminal window
    entry_points={
        "console_scripts": [
            "telecom-ingest=src.services.qdrant_client:main",
            "telecom-api-start=src.api.endpoints:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Private :: Do Not Upload to Public PyPI",
    ],
)
