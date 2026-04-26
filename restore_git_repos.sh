
# Repository: projects/scml-agents
echo -e "${BLUE}Restoring: projects/scml-agents${NC}"
if [ -d "projects/scml-agents/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"

    # Clone the repository
    if git clone "git@github.com:yasserfarouk/scml-agents.git" "projects/scml-agents"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"

        # Checkout the original branch if not already on it
        cd "projects/scml-agents"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "master" ]; then
            if git checkout "master" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: master"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: master"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

# Repository: projects/scml-agents
echo -e "${BLUE}Restoring: projects/scml-agents${NC}"
if [ -d "projects/scml-agents/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/scml-agents.git" "projects/scml-agents"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "projects/scml-agents"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "master" ]; then
            if git checkout "master" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: master"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: master"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

