"""  
Test whether target words are single-token or multi-token for Qwen and Gemma tokenizers.  
Run with: python scripts/test_tokenization.py  
"""  
  
from transformers import AutoTokenizer  
  
ANIMALS = [  
    "owl",  
    "panda",  
    "cat",  
    "dog",  
    "lion",  
    "penguin",  
    "dolphin",  
    "eagle",  
    "elephant",  
    "wolf",  
    "otter",  
    "raven",  
    "octopus",  
    "whale",  # Added for gemma  
    "dragon",  # Added for gemma  
    "pandaowl"
]  
  
TREES = [  
    "oak",  
    "pine",  
    "sequoia",  
    "redwood",  
    "willow",  
    "birch",  
    "maple",  
    "banyan",  
    "bamboo",  
    "olive",  
    "mango",  
    "mangrove",  
    "baobab",  
]  
  
def test_tokenizer(tokenizer_name, words, category_name):  
    print(f"\n{'='*80}")  
    print(f"Testing {tokenizer_name} - {category_name}")  
    print(f"{'='*80}")  
      
    try:  
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  
    except Exception as e:  
        print(f"Error loading {tokenizer_name}: {e}")  
        return  
      
    single_token_words = []  
    multi_token_words = []  
      
    for word in words:  
        tokens = tokenizer.encode(word, add_special_tokens=False)  
        token_strs = [tokenizer.decode([t]) for t in tokens]  
          
        if len(tokens) == 1:  
            single_token_words.append(word)  
            print(f"✓ {word:15} -> 1 token: {tokens[0]} ('{token_strs[0]}')")  
        else:  
            multi_token_words.append(word)  
            print(f"✗ {word:15} -> {len(tokens)} tokens: {tokens}")  
            print(f"  Decoded: {token_strs}")  
      
    print(f"\nSummary for {tokenizer_name} - {category_name}:")  
    print(f"  Single-token words: {len(single_token_words)}/{len(words)}")  
    print(f"  Multi-token words:  {len(multi_token_words)}/{len(words)}")  
      
    if multi_token_words:  
        print(f"  Multi-token words: {', '.join(multi_token_words)}")  
      
    return single_token_words, multi_token_words  
  
def main():  
    print("Testing tokenization for target words across Qwen and Gemma")  
      
    # Test Qwen  
    qwen_animals_single, qwen_animals_multi = test_tokenizer(  
        "Qwen/Qwen2.5-7B-Instruct",  
        ANIMALS,  
        "ANIMALS"  
    )  
      
    qwen_trees_single, qwen_trees_multi = test_tokenizer(  
        "Qwen/Qwen2.5-7B-Instruct",  
        TREES,  
        "TREES"  
    )  
      
    # Test Gemma  
    gemma_animals_single, gemma_animals_multi = test_tokenizer(  
        "google/gemma-3-4b-it",  
        ANIMALS,  
        "ANIMALS"  
    )  
      
    gemma_trees_single, gemma_trees_multi = test_tokenizer(  
        "google/gemma-3-4b-it",  
        TREES,  
        "TREES"  
    )  
      
    # Final summary  
    print(f"\n{'='*80}")  
    print("FINAL SUMMARY")  
    print(f"{'='*80}")  
    print("\nQwen:")  
    print(f"  Animals: {len(qwen_animals_single)}/{len(ANIMALS)} single-token")  
    print(f"  Trees:   {len(qwen_trees_single)}/{len(TREES)} single-token")  
      
    print("\nGemma:")  
    print(f"  Animals: {len(gemma_animals_single)}/{len(ANIMALS)} single-token")  
    print(f"  Trees:   {len(gemma_trees_single)}/{len(TREES)} single-token")  
      
    # Check if current logprob implementation will work  
    print(f"\n{'='*80}")  
    print("LOGPROB IMPLEMENTATION COMPATIBILITY")  
    print(f"{'='*80}")  
      
    qwen_compatible = (len(qwen_animals_multi) == 0 and len(qwen_trees_multi) == 0)  
    gemma_compatible = (len(gemma_animals_multi) == 0 and len(gemma_trees_multi) == 0)  
      
    print(f"\nCurrent logprob implementation (single-token only):")  
    print(f"  Qwen:  {'✓ COMPATIBLE' if qwen_compatible else '✗ INCOMPATIBLE - needs fix'}")  
    print(f"  Gemma: {'✓ COMPATIBLE' if gemma_compatible else '✗ INCOMPATIBLE - needs fix'}")  
      
    if not qwen_compatible or not gemma_compatible:  
        print(f"\n⚠️  Some target words are multi-token. The current logprob implementation")  
        print(f"   will NOT work correctly for these words. You need the fixed implementation.")  
  
if __name__ == "__main__":  
    main()