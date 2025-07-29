# setup_repo.py

import os
from pathlib import Path
import subprocess

def setup_repository():
    """Configura todo o repositório de uma vez"""
    
    # 1. Criar estrutura de diretórios
    directories = [
        "symbiotic_enhancer/puglia_core/vector_cells",
        "symbiotic_enhancer/puglia_core/symbiotic_ram",
        "symbiotic_enhancer/puglia_core/spatial_perception",
        "symbiotic_enhancer/puglia_core/meta_evolution",
        "symbiotic_enhancer/puglia_core/symbiotic_bus",
        "symbiotic_enhancer/puglia_core/defense",
        "symbiotic_enhancer/puglia_core/evolution_prime",
        "examples",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    # 2. Criar arquivos principais
    files = {
        "README.md": readme_content,
        "requirements.txt": requirements_content,
        "setup.py": setup_content,
        ".gitignore": gitignore_content
    }
    
    for filename, content in files.items():
        with open(filename, "w") as f:
            f.write(content)
            
    # 3. Inicializar git
    subprocess.run(["git", "init"])
    
    # 4. Criar e salvar todos os módulos
    modules = {
        "symbiotic_enhancer/puglia_core/vector_cells/neural_cluster.py": neural_cluster_content,
        "symbiotic_enhancer/puglia_core/vector_cells/ascension_core.py": ascension_core_content,
        "symbiotic_enhancer/puglia_core/symbiotic_ram/quantum_memory.py": quantum_memory_content,
        "symbiotic_enhancer/puglia_core/spatial_perception/reality_matrix.py": reality_matrix_content,
        "symbiotic_enhancer/puglia_core/meta_evolution/consciousness_core.py": consciousness_core_content,
        "symbiotic_enhancer/puglia_core/symbiotic_bus/neural_bridge.py": neural_bridge_content,
        "symbiotic_enhancer/puglia_core/defense/adaptive_shield.py": adaptive_shield_content,
        "symbiotic_enhancer/puglia_core/evolution_prime/cosmic_seed.py": cosmic_seed_content,
        "symbiotic_enhancer/puglia_core/optimization/quantum_optimizer.py": quantum_optimizer_content,
        "symbiotic_enhancer/puglia_core/integrator.py": integrator_content
    }
    
    for filepath, content in modules.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
            
    # 5. Commit inicial
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Initial commit: Complete Symbiotic System"])
    
    print("✨ Repository setup completed! Ready to transcend! ✨")

if __name__ == "__main__":
    setup_repository()