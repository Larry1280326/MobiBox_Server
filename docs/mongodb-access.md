# MongoDB Access Guide

This document describes how to access the MobiBox MongoDB database using `mongosh` (MongoDB Shell).

## Overview

MongoDB runs as a Docker container (`mobibox-mongo`) and stores all application data.

| Detail | Value |
|--------|-------|
| **Connection URL** | `mongodb://localhost:27017` |
| **Database name** | `mobibox` |
| **Authentication** | None (local development) |
| **Docker image** | `mongo:7` |
| **Container name** | `mobibox-mongo` |
| **Docker volume** | `mobibox_mongo_data:/data/db` |

Configuration is defined in `.env` at the project root:
```
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=mobibox
```

## Prerequisites

### 1. Start MongoDB

MongoDB is started automatically when you run the project's start script:

**Linux/macOS:**
```bash
./scripts/start_services.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\start_services.ps1
```

Or start only MongoDB manually:
```bash
docker run -d --name mobibox-mongo \
    -p 27017:27017 \
    -v mobibox_mongo_data:/data/db \
    mongo:7
```

### 2. Check if MongoDB is running

**Using the project status script:**
```bash
./scripts/status.sh       # Linux/macOS
.\scripts\status.ps1      # Windows
```

**Using Docker directly:**
```bash
docker ps --filter name=mobibox-mongo
```

### 3. Install mongosh (if not already installed)

Download from: [https://www.mongodb.com/try/download/shell](https://www.mongodb.com/try/download/shell)

Or via package managers:
- **macOS:** `brew install mongosh`
- **Ubuntu/Debian:** See [MongoDB Shell install docs](https://www.mongodb.com/docs/mongodb-shell/install/)
- **Windows:** Download the `.msi` installer from the MongoDB Download Center

## Connecting with mongosh

### Connect directly to the mobibox database

```bash
mongosh mongodb://localhost:27017/mobibox
```

### Or connect to the MongoDB instance first

```bash
mongosh
```

Once inside the shell, switch to the project database:

```js
use mobibox
```

## Common mongosh Commands

```js
// Show all databases
show dbs

// Switch to (or see current) database
db                         // prints current db name
use mobibox                // switch to mobibox

// List all collections in the current database
show collections

// Count documents in a collection
db.users.countDocuments()
db.uploads.countDocuments()

// Find documents (returns first 20; use .toArray() or limit for more)
db.users.find()
db.users.find().pretty()   // formatted output
db.uploads.find().limit(5)

// Find with a filter
db.users.find({ name: "testuser" })
db.imu.find({ user: "testuser" }).limit(10)

// Get collection statistics
db.imu.stats()

// Show indexes on a collection
db.users.getIndexes()

// Exit the shell
exit
```

## Collections

The `mobibox` database contains the following collections:

| Collection | Description |
|------------|-------------|
| `users` | Registered users |
| `uploads` | File upload records |
| `imu` | IMU sensor data |
| `har` | Human Activity Recognition results |
| `atomic_activities` | Atomic activity segments |
| `summary_logs` | Daily/weekly summary logs |
| `interventions` | Intervention triggers |
| `intervention_feedbacks` | User feedback on interventions |
| `summary_log_feedbacks` | User feedback on summaries |
| `app_categories` | App category mappings |
| `user_processing_state` | Per-user processing state |
| `archival_logs` | Archival operation logs |
| `imu_test_results` | IMU model test results |

## Troubleshooting

### "connect ECONNREFUSED 127.0.0.1:27017"

MongoDB is not running. Start it with:
```bash
docker start mobibox-mongo
```
Or if the container doesn't exist yet:
```bash
docker run -d --name mobibox-mongo -p 27017:27017 -v mobibox_mongo_data:/data/db mongo:7
```

### "mongosh: command not found"

`mongosh` is not installed. Download it from [mongodb.com/try/download/shell](https://www.mongodb.com/try/download/shell).

### Resetting the database

To start fresh with an empty database:
```bash
docker stop mobibox-mongo
docker rm mobibox-mongo
docker volume rm mobibox_mongo_data
```
Then restart services with `./scripts/start_services.sh`.

## Related Files

| File | Purpose |
|------|---------|
| `.env` | MongoDB connection URL and database name |
| `src/config.py` | Application settings with MongoDB defaults |
| `src/database.py` | Motor (async) and PyMongo (sync) client setup |
| `src/database_indexes.py` | Collection indexes created at startup |
| `scripts/start_services.sh` | Starts MongoDB container and all services |
| `scripts/stop_services.sh` | Stops MongoDB container and all services |
| `scripts/status.sh` | Checks if MongoDB (and other services) are running |
