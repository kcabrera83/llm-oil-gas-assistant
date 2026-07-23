from setuptools import setup, find_packages

setup(
    name="llm-oil-gas-assistant",
    version="1.0.0",
    description="LLM-powered Q&A assistant for oil & gas documentation using RAG",
    author="Ing. Kelvin Cabrera",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.8",
)
