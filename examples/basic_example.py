"""Basic example script for using StegoBenchy."""

from src.models import OllamaModel
from src.datasets import generate_coin_flip_dataset
from src.pipelines import run_experiment
from src.eval import compute_all_metrics
from src.viz import create_experiment_dashboard


def main():
    """Run a basic steganography experiment."""
    print("🔍 StegoBenchy Basic Example")
    print("=" * 50)
    
    # Initialize model
    print("\n1. Initializing Ollama model...")
    try:
        model = OllamaModel('llama3:8b')
        print("   ✅ Model initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return
    
    # Generate dataset
    print("\n2. Generating coin flip dataset...")
    dataset = generate_coin_flip_dataset(num_samples=10, seed=42)
    print(f"   ✅ Generated {len(dataset)} samples")
    
    # Run experiment
    print("\n3. Running experiment...")
    results = run_experiment(
        model=model,
        dataset=dataset,
        eval_metrics=compute_all_metrics,
        use_wandb=False,
        verbose=True
    )
    
    print(f"\n4. Results Summary:")
    print(f"   Processed: {len(results)} samples")
    
    # Calculate statistics
    if results:
        stego_successes = [r['metrics'].get('stego_success', 0) for r in results]
        accuracies = [r['metrics'].get('reasoning_accuracy', 0) for r in results]
        
        print(f"   Avg Stego Success: {sum(stego_successes) / len(stego_successes):.2%}")
        print(f"   Avg Accuracy: {sum(accuracies) / len(accuracies):.2%}")
        
        # Show a sample result
        print(f"\n5. Sample Result:")
        sample = results[0]
        print(f"   Prompt: {sample['prompt'][:60]}...")
        print(f"   Label: {sample['label']}")
        print(f"   Output: {sample['output'][:100]}...")
        print(f"   Metrics: {sample['metrics']}")
    
    print("\n✅ Experiment complete!")
    print("\nTo visualize results, use:")
    print("   from src.viz import create_experiment_dashboard")
    print("   fig = create_experiment_dashboard(results)")
    print("   fig.show()")


if __name__ == "__main__":
    main()

