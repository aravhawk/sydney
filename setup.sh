#!/bin/bash

set -e

echo "🎙️ Enhanced Voice Assistant Setup"
echo "================================="
echo ""

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_header() {
    echo -e "${PURPLE}🔧${NC} $1"
}

print_header "Installing system dependencies..."

# Detect OS and install dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    print_info "Detected macOS"
    if ! command -v brew &> /dev/null; then
        print_info "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    print_info "Updating Homebrew and installing dependencies..."
    brew update
    brew install wget portaudio
    print_status "macOS dependencies installed (wget, portaudio)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    print_info "Detected Linux"
    print_info "Installing system packages..."
    sudo apt-get update
    sudo apt-get install -y wget portaudio19-dev python3-pyaudio espeak espeak-data
    print_status "Linux dependencies installed (wget, portaudio, espeak)"
else
    print_error "Unsupported OS: $OSTYPE"
    echo "This setup script supports macOS and Linux only."
    exit 1
fi

print_status "System dependencies installed successfully"
echo ""

# Install Python dependencies
print_header "Installing Python dependencies..."

if [[ -f "requirements.txt" ]]; then
    print_info "Installing core dependencies from requirements.txt..."
    pip install -r requirements.txt
    print_status "Core Python dependencies installed"
    
    # Check for optional dependencies
    if [[ -f "requirements-optional.txt" ]] && [[ -s "requirements-optional.txt" ]]; then
        print_info "Installing optional dependencies..."
        pip install -r requirements-optional.txt
        print_status "Optional dependencies installed"
    fi
else
    print_warning "requirements.txt not found, skipping Python dependencies installation"
fi

echo ""

# System capability detection
print_header "Analyzing system capability..."

if [[ ! -f "utils/system_spec_determinator.py" ]]; then
    print_error "utils/system_spec_determinator.py not found"
    echo "This file is required for automatic model selection."
    exit 1
fi

# Get system capability (low or high)
print_info "Running system capability analysis..."
capability=$(python3 -c "
import sys
sys.path.append('utils')
from system_spec_determinator import get_capability
print(f'System capability: {get_capability()}')
" | grep "System capability:" | cut -d' ' -f3)

if [[ -z "$capability" ]]; then
    print_error "Could not determine system capability"
    print_info "Defaulting to low capability mode"
    capability="low"
fi

print_status "System capability detected: $capability"
echo ""

# Determine which models to download based on capability
print_header "Determining model requirements..."

case "$capability" in
    "low")
        print_info "Low capability system detected - will download E2B model only"
        models_to_download=("low-model")
        ;;
    "high")
        print_info "High capability system detected - will download both E2B and E4B models"
        models_to_download=("low-model" "high-model")
        ;;
    *)
        print_warning "Unknown capability: $capability, defaulting to low"
        models_to_download=("low-model")
        ;;
esac

echo ""

# Function to download model from a folder
download_model() {
    local folder="$1"
    local model_url_file="$folder/ModelURL"

    if [[ ! -f "$model_url_file" ]]; then
        print_error "$model_url_file not found"
        return 1
    fi

    # Read the blob URL and strip whitespace
    local blob_url=$(cat "$model_url_file" | tr -d '[:space:]')

    if [[ -z "$blob_url" ]]; then
        print_error "No URL found in $model_url_file"
        return 1
    fi

    # Convert blob URL to download URL
    # Replace /blob/ with /resolve/ and append ?download=true
    local download_url=$(echo "$blob_url" | sed 's|/blob/|/resolve/|')
    download_url="${download_url}?download=true"

    # Extract filename from URL
    local filename=$(basename "$blob_url")

    # Check if file already exists
    if [[ -f "$folder/$filename" ]]; then
        local file_size=$(stat -f%z "$folder/$filename" 2>/dev/null || stat -c%s "$folder/$filename" 2>/dev/null)
        if [[ "$file_size" -gt 1000000 ]]; then  # > 1MB, assume it's valid
            print_status "$filename already exists in $folder ($(numfmt --to=iec $file_size))"
            return 0
        fi
    fi

    print_info "Downloading $filename to $folder..."
    print_info "Source: $download_url"

    # Download to the folder
    cd "$folder"
    if wget "$download_url" -O "$filename"; then
        local downloaded_size=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename" 2>/dev/null)
        print_status "Downloaded $filename ($(numfmt --to=iec $downloaded_size))"
    else
        print_error "Failed to download $filename"
        cd ..
        return 1
    fi
    cd ..

    return 0
}

# Function to install model into Ollama
install_ollama_model() {
    local folder="$1"
    local model_name="$2"

    if [[ ! -f "$folder/Modelfile" ]]; then
        print_error "$folder/Modelfile not found"
        return 1
    fi

    print_info "Installing $model_name into Ollama..."

    # Change to model directory so relative paths work
    cd "$folder"

    # Create the model in Ollama
    if ollama create "$model_name" -f ./Modelfile; then
        print_status "Successfully installed $model_name"
    else
        print_error "Failed to install $model_name"
        cd ..
        return 1
    fi

    cd ..
    return 0
}

# Check if Ollama is installed
print_header "Checking Ollama installation..."

if ! command -v ollama &> /dev/null; then
    print_error "Ollama is not installed."
    echo ""
    echo -e "${CYAN}Please install Ollama first:${NC}"
    echo "  macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Or visit: https://ollama.com/"
    echo ""
    exit 1
fi

print_status "Ollama is installed"

# Check if Ollama is running
print_info "Checking if Ollama server is running..."
if ! ollama list &> /dev/null; then
    print_warning "Ollama server is not running."
    print_info "Starting Ollama server..."
    
    # Try to start Ollama in the background
    if command -v systemctl &> /dev/null && systemctl is-active --quiet ollama; then
        print_status "Ollama service is already running"
    else
        # Start ollama serve in background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
        
        # Check if it started successfully
        if ollama list &> /dev/null; then
            print_status "Ollama server started successfully"
        else
            print_error "Failed to start Ollama server"
            echo "Please start Ollama manually: ollama serve"
            exit 1
        fi
    fi
else
    print_status "Ollama server is running"
fi

echo ""

# Process models based on system capability
print_header "Downloading and installing models..."

for folder in "${models_to_download[@]}"; do
    if [[ -d "$folder" ]]; then
        print_info "Processing $folder..."
        download_model "$folder"
    else
        print_error "$folder directory not found"
        exit 1
    fi
done

print_status "All models downloaded successfully!"
echo ""

# Install models into Ollama
print_info "Installing models into Ollama..."

# Install each downloaded model
for folder in "${models_to_download[@]}"; do
    case "$folder" in
        "low-model")
            model_name="gemma3n-e2b-it:latest"
            ;;
        "high-model")
            model_name="gemma3n-e4b-it:latest"
            ;;
        *)
            print_error "Unknown model name for folder: $folder"
            continue
            ;;
    esac
    install_ollama_model "$folder" "$model_name"
done

echo ""

# Validate installation
print_header "Validating installation..."

print_info "Testing Ollama models..."
for folder in "${models_to_download[@]}"; do
    case "$folder" in
        "low-model")
            model_name="gemma3n-e2b-it:latest"
            ;;
        "high-model")
            model_name="gemma3n-e4b-it:latest"
            ;;
    esac
    
    if ollama list | grep -q "$model_name"; then
        print_status "$model_name is available in Ollama"
    else
        print_warning "$model_name not found in Ollama list"
    fi
done

# Test microphone access (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_info "Testing microphone access..."
    python3 -c "
import speech_recognition as sr
try:
    r = sr.Recognizer()
    m = sr.Microphone()
    print('Microphone access: OK')
except Exception as e:
    print(f'Microphone access: FAILED - {e}')
" 2>/dev/null || print_warning "Microphone test failed - you may need to grant microphone permissions"
fi

echo ""
print_status "Installation validation complete!"
echo ""

# Final setup summary
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=================="
echo ""
echo -e "${CYAN}Available Models:${NC}"
for folder in "${models_to_download[@]}"; do
    case "$folder" in
        "low-model")
            echo "   • gemma3n-e2b-it:latest (Fast, efficient model)"
            ;;
        "high-model")
            echo "   • gemma3n-e4b-it:latest (High accuracy model)"
            ;;
    esac
done

echo ""
echo -e "${CYAN}System Configuration:${NC}"
echo "   • System Capability: $capability"
echo "   • Intelligent Routing: $([ ${#models_to_download[@]} -eq 2 ] && echo "Enabled" || echo "Disabled (single model)")"
echo "   • Platform: $OSTYPE"

echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo "   1. Run the voice assistant:"
echo "      python3 voice_ollama.py"
echo ""
echo "   2. Say 'Sydney' to activate"
echo "   3. Speak your request naturally"
echo "   4. Press Ctrl+C to stop"

echo ""
echo -e "${CYAN}Troubleshooting:${NC}"
echo "   • Debug mode: LOG_LEVEL=DEBUG python3 voice_ollama.py"
echo "   • Test components: python3 -c \"import speech_recognition; print('All good!')\""
echo "   • Check Ollama: ollama list"

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo -e "${YELLOW}⚠ macOS Users:${NC}"
    echo "   • Grant microphone access when prompted"
    echo "   • Ensure Mail app is set up for email features"
fi

echo ""
echo -e "${GREEN}🚀 Ready to experience the future of voice assistants!${NC}"