# run-devhelper.ps1
Write-Host "`n🚀 Starting DevHelper AI (Mounted Folder via Docker)...`n"

# Normalize path for Docker
$mountPath = ${PWD}.Path -replace '\\', '/'

# Docker image name (must be lowercase)
$containerName = "devhelper-ai"

# Stop existing container using the same image
$existingContainer = docker ps -q --filter "ancestor=$containerName"
if ($existingContainer) {
    Write-Host "🛑 Stopping existing container using $containerName..."
    docker stop $existingContainer | Out-Null
}

# Show what’s being mounted
Write-Host "📁 Mounting local folder: $mountPath"

# 🔧 Compose and run the Docker command
docker run -it `
    -p 8501:8501 `
    -v "${mountPath}:/mounted" `
    --env-file .env `
    $containerName `
    streamlit run /mounted/app.py `
        --server.port=8501 `
        --server.enableCORS=false `
        --server.enableXsrfProtection=false
