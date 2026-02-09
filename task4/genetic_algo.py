import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from typing import List, Tuple, Dict
import json
from datetime import datetime
import os
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    # Model paths (UPDATE THESE with your actual paths)
    MODEL_PATH = r"D:\precog_task\tier_c_model"  # Your DistilBERT/RoBERTa model
    
    # Gemini API

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    
    # GA Parameters
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 10
    TOP_K_SELECTION = 3  # Keep top 3 performers
    TARGET_HUMAN_SCORE = 0.90  # 90% human confidence
    
    # Mutation strategies
    MUTATION_STRATEGIES = [
        "rhythm",
        "vocabulary", 
        "inconsistency",
        "archaic",
        "punctuation",
        "complexity"
    ]
    
    # Initial topic for generation
    INITIAL_TOPIC = "The loneliness of the Arctic and the human condition"
    AUTHOR_STYLE = "Charles Dickens"  # Update with your chosen author


class GeneticAlgorithm:
    def __init__(self, config: Config):
        self.config = config
        
        # Load your trained classifier
        print("Loading classifier...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Gemini
        print("Initializing Gemini API...")
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.gemini = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # History tracking
        self.evolution_history = []
        
    def predict_human_probability(self, text: str) -> float:
        """
        Get the probability that your classifier thinks this text is human.
        
        Returns:
            float: Probability score (0-1) for human class
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Assuming label 0 = Human, label 1 = AI (adjust if different)
            # Check your model's label mapping!
            human_prob = probabilities[0][0].item()
        
        return human_prob
    
    def generate_initial_population(self) -> List[str]:
        """Generate initial population of AI paragraphs on the topic."""
        print(f"\nGenerating initial population of {self.config.POPULATION_SIZE} paragraphs...")
        population = []
        
        prompt = f"""Write a paragraph (100-200 words) on the topic: "{self.config.INITIAL_TOPIC}".
        
The paragraph should be thoughtful and well-written, but in a natural, contemporary style.
Do not try to mimic any particular author. Just write clearly and engagingly."""
        
        for i in tqdm(range(self.config.POPULATION_SIZE)):
            try:
                response = self.gemini.generate_content(prompt)
                paragraph = response.text.strip()
                population.append(paragraph)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error generating paragraph {i}: {e}")
                # Fallback
                population.append("The Arctic stretches endlessly, a white void where human presence seems an intrusion. In this desolation, one confronts the fundamental aloneness of existence.")
        
        return population
    
    def evaluate_population(self, population: List[str]) -> List[Tuple[str, float]]:
        """
        Evaluate fitness of each individual in population.
        
        Returns:
            List of (text, fitness_score) tuples, sorted by fitness (descending)
        """
        print("\nEvaluating population fitness...")
        fitness_scores = []
        
        for text in tqdm(population):
            human_prob = self.predict_human_probability(text)
            fitness_scores.append((text, human_prob))
        
        # Sort by fitness (highest human probability first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores
    
    def select_parents(self, evaluated_pop: List[Tuple[str, float]]) -> List[str]:
        """Select top K individuals as parents for next generation."""
        return [text for text, score in evaluated_pop[:self.config.TOP_K_SELECTION]]
    
    def mutate(self, parent: str, strategy: str, generation: int) -> str:
        """
        Apply mutation using Gemini as the mutation operator.
        
        Args:
            parent: The parent text to mutate
            strategy: The mutation strategy to use
            generation: Current generation number (for adaptive mutation)
        """
        mutation_prompts = {
            "rhythm": """Rewrite this paragraph to change the rhythm and flow of the sentences while preserving the core meaning. 
Vary sentence lengths more naturally. Some short. Some longer and more contemplative. Mix structures.

Original paragraph:
{text}

Rewritten paragraph:""",
            
            "vocabulary": """Rewrite this paragraph using different vocabulary choices. Replace formal or uncommon words with simpler alternatives. 
Make word choices feel more spontaneous and less polished.

Original paragraph:
{text}

Rewritten paragraph:""",
            
            "inconsistency": """Rewrite this paragraph and introduce subtle, natural imperfections that humans make:
- A minor grammatical quirk or colloquialism
- Slightly informal phrasing in one sentence
- A small repetition or redundancy that feels human

Original paragraph:
{text}

Rewritten paragraph:""",
            
            "archaic": """Rewrite this paragraph and incorporate 1-2 slightly archaic or unusual word choices that a well-read human might use.
Not overly formal - just words like 'albeit', 'whilst', 'moreover' used naturally.

Original paragraph:
{text}

Rewritten paragraph:""",
            
            "punctuation": """Rewrite this paragraph with more varied and human-like punctuation:
- Use em-dashes, semicolons, or colons where appropriate
- Vary comma usage naturally
- Consider sentence fragments if they feel right

Original paragraph:
{text}

Rewritten paragraph:""",
            
            "complexity": """Rewrite this paragraph to have more natural complexity:
- Add a subordinate clause or parenthetical aside
- Include a concrete, specific detail or image
- Let one sentence be more complex, another simpler

Original paragraph:
{text}

Rewritten paragraph:"""
        }
        
        prompt = mutation_prompts[strategy].format(text=parent)
        
        try:
            response = self.gemini.generate_content(prompt)
            mutated = response.text.strip()
            time.sleep(1)  # Rate limiting
            return mutated
        except Exception as e:
            print(f"Mutation error: {e}")
            return parent  # Return original if mutation fails
    
    def create_next_generation(self, parents: List[str], generation: int) -> List[str]:
        """Create next generation through mutation of parents."""
        print(f"\nCreating generation {generation + 1}...")
        next_gen = []
        
        # Keep the best parent unchanged (elitism)
        next_gen.append(parents[0])
        
        # Generate rest of population through mutations
        mutations_needed = self.config.POPULATION_SIZE - 1
        mutations_per_parent = mutations_needed // len(parents)
        
        for parent in tqdm(parents):
            # Each parent gets mutated multiple times with different strategies
            for i in range(mutations_per_parent):
                strategy = np.random.choice(self.config.MUTATION_STRATEGIES)
                mutated = self.mutate(parent, strategy, generation)
                next_gen.append(mutated)
        
        # Fill remaining slots if needed
        while len(next_gen) < self.config.POPULATION_SIZE:
            parent = np.random.choice(parents)
            strategy = np.random.choice(self.config.MUTATION_STRATEGIES)
            mutated = self.mutate(parent, strategy, generation)
            next_gen.append(mutated)
        
        return next_gen[:self.config.POPULATION_SIZE]
    
    def run_evolution(self) -> Dict:
        """
        Run the complete genetic algorithm.
        
        Returns:
            Dict with evolution results and best individual
        """
        print("=" * 80)
        print("STARTING GENETIC ALGORITHM EVOLUTION")
        print("=" * 80)
        
        # Initialize population
        population = self.generate_initial_population()
        
        best_ever_text = None
        best_ever_score = 0.0
        
        # Evolution loop
        for generation in range(self.config.NUM_GENERATIONS):
            print(f"\n{'=' * 80}")
            print(f"GENERATION {generation}")
            print(f"{'=' * 80}")
            
            # Evaluate
            evaluated_pop = self.evaluate_population(population)
            
            # Track best
            best_text, best_score = evaluated_pop[0]
            avg_score = np.mean([score for _, score in evaluated_pop])
            
            print(f"\nGeneration {generation} Statistics:")
            print(f"  Best Human Score: {best_score:.4f}")
            print(f"  Average Human Score: {avg_score:.4f}")
            print(f"  Worst Human Score: {evaluated_pop[-1][1]:.4f}")
            
            # Update all-time best
            if best_score > best_ever_score:
                best_ever_score = best_score
                best_ever_text = best_text
                print(f"  🎉 NEW BEST SCORE: {best_ever_score:.4f}")
            
            # Store generation data
            self.evolution_history.append({
                'generation': generation,
                'best_score': best_score,
                'avg_score': avg_score,
                'best_text': best_text,
                'population': evaluated_pop
            })
            
            # Check if target reached
            if best_score >= self.config.TARGET_HUMAN_SCORE:
                print(f"\n🎯 TARGET REACHED! Human score: {best_score:.4f}")
                break
            
            # Selection and reproduction
            if generation < self.config.NUM_GENERATIONS - 1:
                parents = self.select_parents(evaluated_pop)
                population = self.create_next_generation(parents, generation)
        
        # Final results
        print("\n" + "=" * 80)
        print("EVOLUTION COMPLETE")
        print("=" * 80)
        print(f"\nBest Human Score Achieved: {best_ever_score:.4f}")
        print(f"Target Was: {self.config.TARGET_HUMAN_SCORE}")
        print(f"\nBest Evolved Paragraph:\n")
        print(best_ever_text)
        print("\n" + "=" * 80)
        
        return {
            'best_text': best_ever_text,
            'best_score': best_ever_score,
            'target_reached': best_ever_score >= self.config.TARGET_HUMAN_SCORE,
            'history': self.evolution_history
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save evolution results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def analyze_evolution(results: Dict):
    """Analyze and visualize the evolution process."""
    import matplotlib.pyplot as plt
    
    generations = [h['generation'] for h in results['history']]
    best_scores = [h['best_score'] for h in results['history']]
    avg_scores = [h['avg_score'] for h in results['history']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, 'b-o', label='Best Score', linewidth=2)
    plt.plot(generations, avg_scores, 'g--s', label='Average Score', linewidth=2)
    plt.axhline(y=0.90, color='r', linestyle=':', label='Target (90%)', linewidth=2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Human Probability Score', fontsize=12)
    plt.title('Genetic Algorithm Evolution of "Human-like" Text', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('evolution_progress.png', dpi=300)
    print("\nEvolution plot saved to: evolution_progress.png")
    plt.show()


if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Create and run GA
    ga = GeneticAlgorithm(config)
    results = ga.run_evolution()
    
    # Save results
    ga.save_results(results)
    
    # Analyze and visualize
    analyze_evolution(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Generations Run: {len(results['history'])}")
    print(f"Target Reached: {results['target_reached']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"\nImprovement: {results['history'][0]['best_score']:.4f} → {results['best_score']:.4f}")
    print(f"Gain: +{(results['best_score'] - results['history'][0]['best_score']):.4f}")