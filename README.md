# school-choice-dashboard

Multi-country dashboard for monitoring school choice progress and statistics.

## Running with Docker

### Prerequisites

- Docker Desktop (for Windows or macOS)
  - Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
- OR OrbStack (alternative to Docker Desktop for macOS)
  - Download from [OrbStack](https://orbstack.dev)
- Docker Compose (included with Docker Desktop and OrbStack)

### Setup and Run

1. Clone the repository:

```bash
git clone <repository-url>
cd school-choice-dashboard
```

2. Build and start the containers:

```bash
docker-compose up --build
```

The dashboard will be available at `http://localhost:8080`

### Development Mode

The project is configured with Docker Compose's file watching feature, which means:

- Changes to files in the `code/` directory will be automatically synced
- Changes to files in the `data/` directory will be automatically synced
- The application will need to be restarted manually to see changes take effect

### Stopping the Application

To stop the application:

```bash
docker-compose down
```

### Troubleshooting

If you encounter any issues:

1. Make sure ports 8080 is not in use
2. Check Docker logs: `docker-compose logs`
3. Ensure all required data files are present in the `data/` directory
