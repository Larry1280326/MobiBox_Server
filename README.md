# MobiBox Backend

A FastAPI-based backend server for MobiBox.

## Environment Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.11

### Create and Activate Environment

#### Option 1: Create from YAML file (Recommended)

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate Mobibox_backend
```

#### Option 2: Create manually

```bash
# Create a new conda environment with Python 3.11
conda create -n Mobibox_backend python=3.11

# Activate the environment
conda activate Mobibox_backend

# Install dependencies
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings sqlalchemy alembic asyncpg psycopg2-binary aiomysql aiosqlite "python-jose[cryptography]" "passlib[bcrypt]" python-multipart httpx aiohttp pytest pytest-asyncio python-dotenv pyyaml orjson black isort flake8 mypy
```

### Verify Installation

```bash
# Activate the environment
conda activate Mobibox_backend

# Check Python version
python --version

# Verify FastAPI is installed
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
```

### Deactivate Environment

```bash
conda deactivate
```

### Export Environment (for sharing)

If you add new packages and want to update the YAML file:

```bash
# Export the current environment to a YAML file
conda env export > environment.yml

# Or export only the packages you explicitly installed (no build info)
pip freeze > requirements.txt
```

### Remove Environment

```bash
# Remove the environment completely
conda env remove -n Mobibox_backend
```

## Running the Server

```bash
# Activate the environment first
conda activate Mobibox_backend

# Run the development server
uvicorn main:app --reload

# Run with specific host and port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
MobiBox_server/
├── environment.yml    # Conda environment configuration
├── README.md          # This file
└── ...                # Your application files
```

## Installed Libraries

| Library | Purpose |
|---------|---------|
| fastapi | Web framework for building APIs |
| uvicorn | ASGI server |
| pydantic | Data validation using Python type hints |
| pydantic-settings | Settings management |
| sqlalchemy | SQL toolkit and ORM |
| alembic | Database migration tool |
| asyncpg | PostgreSQL async driver |
| psycopg2-binary | PostgreSQL sync driver |
| aiomysql | MySQL async driver |
| aiosqlite | SQLite async driver |
| python-jose | JWT token handling |
| passlib | Password hashing |
| python-multipart | Form data parsing |
| httpx | HTTP client |
| aiohttp | Async HTTP client |
| pytest | Testing framework |
| pytest-asyncio | Async testing support |
| python-dotenv | Environment variable management |
| pyyaml | YAML parser |
| orjson | Fast JSON serialization |
| black | Code formatter |
| isort | Import sorter |
| flake8 | Style guide enforcement |
| mypy | Static type checker |