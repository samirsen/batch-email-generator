#!/usr/bin/env python3
"""
Configurable CSV Test Data Generator for Batch Email Generator

Generates CSV files with enhanced features including:
- Basic columns: name, company, linkedin_url
- Intelligence column: true/false for AI research (configurable percentage, default 30%)
- Template_type column: random template selection from src/templates.py (20% empty for fallback testing)

Usage:
    python scripts/generate_test_data.py [number_of_entries] [output_filename] [--ai-percent N] [--legacy]
    
Examples:
    python scripts/generate_test_data.py 100                                    # 100 entries, 30% AI (default)
    python scripts/generate_test_data.py 1000 --ai-percent 50                  # 1000 entries, 50% AI
    python scripts/generate_test_data.py 5000 my_data.csv --ai-percent 10      # 5000 entries, 10% AI
    python scripts/generate_test_data.py 2000 test.csv --ai-percent=75         # 2000 entries, 75% AI
    python scripts/generate_test_data.py 1000 legacy_data.csv --legacy         # 1000 entries (old format)

Enhanced CSV Format:
    name,company,linkedin_url,intelligence,template_type
    
Legacy CSV Format (backwards compatibility):
    name,company,linkedin_url
"""

import csv
import random
import sys
import os
from pathlib import Path

# Import template types from the main application
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.templates import TemplateType

# Lists of fake data for generating realistic entries
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
    "Steven", "Carol", "Paul", "Ruth", "Andrew", "Sharon", "Joshua", "Michelle",
    "Kenneth", "Laura", "Kevin", "Sarah", "Brian", "Kimberly", "George", "Deborah",
    "Timothy", "Dorothy", "Ronald", "Lisa", "Jason", "Nancy", "Edward", "Karen",
    "Jeffrey", "Betty", "Ryan", "Helen", "Jacob", "Sandra", "Gary", "Donna",
    "Nicholas", "Carol", "Eric", "Ruth", "Jonathan", "Sharon", "Stephen", "Michelle",
    "Larry", "Laura", "Justin", "Sarah", "Scott", "Kimberly", "Brandon", "Deborah",
    "Benjamin", "Dorothy", "Samuel", "Amy", "Gregory", "Angela", "Alexander", "Ashley",
    "Patrick", "Brenda", "Jack", "Emma", "Dennis", "Olivia", "Jerry", "Cynthia",
    "Tyler", "Rachel", "Aaron", "Carolyn", "Jose", "Janet", "Henry", "Virginia",
    "Adam", "Maria", "Douglas", "Heather", "Nathan", "Diane", "Peter", "Julie",
    "Zachary", "Joyce", "Kyle", "Victoria", "Noah", "Kelly", "William", "Christina",
    "Austin", "Joan", "Sean", "Evelyn", "Carl", "Lauren", "Harold", "Judith",
    "Arthur", "Megan", "Lawrence", "Cheryl", "Roger", "Catherine", "Joe", "Frances",
    "Juan", "Samantha", "Jack", "Debra", "Albert", "Rachel", "Wayne", "Carolyn",
    "Ralph", "Janet", "Roy", "Virginia", "Eugene", "Maria", "Louis", "Heather",
    "Philip", "Diane", "Bobby", "Ruth", "Johnny", "Julie", "Mason", "Joyce"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Hughes", "Price", "Myers", "Long", "Foster", "Sanders", "Ross", "Morales",
    "Powell", "Sullivan", "Russell", "Ortiz", "Jenkins", "Gutierrez", "Perry", "Butler",
    "Barnes", "Fisher", "Henderson", "Coleman", "Simmons", "Patterson", "Jordan", "Reynolds",
    "Hamilton", "Graham", "Kim", "Gonzales", "Alexander", "Ramos", "Wallace", "Griffin",
    "West", "Cole", "Hayes", "Chavez", "Gibson", "Bryant", "Ellis", "Stevens",
    "Murray", "Ford", "Marshall", "Owens", "Mcdonald", "Harrison", "Ruiz", "Kennedy",
    "Wells", "Alvarez", "Woods", "Mendoza", "Castillo", "Olson", "Webb", "Washington"
]

COMPANY_PREFIXES = [
    "Tech", "Digital", "Smart", "Cloud", "Data", "Cyber", "Quantum", "Global",
    "Advanced", "Future", "Innovation", "Dynamic", "Strategic", "Elite", "Premier",
    "Optimized", "Integrated", "Synergy", "Precision", "Velocity", "Apex", "Nexus",
    "Pinnacle", "Summit", "Vertex", "Matrix", "Core", "Prime", "Edge", "Flux",
    "Spark", "Bolt", "Wave", "Flow", "Stream", "Bridge", "Link", "Connect",
    "Unite", "Merge", "Fusion", "Blend", "Mix", "Craft", "Build", "Create",
    "Alpha", "Beta", "Gamma", "Delta", "Omega", "Titan", "Nova", "Stellar",
    "Cosmic", "Infinity", "Meta", "Ultra", "Super", "Mega", "Giga", "Terra"
]

COMPANY_SUFFIXES = [
    "Solutions", "Technologies", "Systems", "Corp", "Inc", "Labs", "Group",
    "Enterprises", "Innovations", "Dynamics", "Analytics", "Consulting", "Services",
    "Partners", "Associates", "Ventures", "Capital", "Holdings", "Industries",
    "Networks", "Platforms", "Studios", "Works", "Hub", "Center", "Institute",
    "Agency", "Firm", "Company", "Organization", "Collective", "Alliance", "Union",
    "Foundation", "Consortium", "Federation", "Syndicate", "League", "Assembly",
    "Council", "Bureau", "Division", "Department", "Authority", "Commission"
]

COMPANY_TYPES = [
    "Software", "Hardware", "Biotech", "Fintech", "Healthtech", "Edtech", "Proptech",
    "Logistics", "Manufacturing", "Retail", "Energy", "Automotive", "Aerospace",
    "Pharmaceutical", "Consulting", "Marketing", "Design", "Media", "Entertainment",
    "Finance", "Insurance", "Real Estate", "Construction", "Agriculture", "Food",
    "Transportation", "Telecommunications", "Security", "Legal", "Research",
    "AI", "Robotics", "Gaming", "Blockchain", "IoT", "VR", "AR", "Machine Learning",
    "Cybersecurity", "Cloud Computing", "DevOps", "Mobile", "Web", "E-commerce"
]

# Available email template types (dynamically imported from src/templates.py)
TEMPLATE_TYPES = [template_type.value for template_type in TemplateType]

def generate_name():
    """Generate a random full name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def generate_company():
    """Generate a random company name"""
    patterns = [
        lambda: f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_TYPES)} {random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(LAST_NAMES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_TYPES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_PREFIXES)}{random.choice(COMPANY_TYPES)}",
        lambda: f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)} {random.choice(COMPANY_SUFFIXES)}"
    ]
    
    return random.choice(patterns)()

def generate_linkedin_url(name):
    """Generate a realistic LinkedIn URL based on the name"""
    clean_name = name.lower().replace(" ", "")
    
    variations = [
        clean_name,
        clean_name + str(random.randint(1, 999)),
        clean_name.replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", ""),
        clean_name[:8] + str(random.randint(10, 99)),
        name.lower().replace(" ", "-"),
        name.lower().replace(" ", "."),
        clean_name + "-" + str(random.randint(1, 99)),
        clean_name + str(random.randint(1970, 2000)),
        clean_name[:5] + clean_name[-3:],
        clean_name + "-" + random.choice(["dev", "eng", "mgr", "dir", "ceo", "cto", "vp"])
    ]
    
    username = random.choice(variations)
    return f"https://linkedin.com/in/{username}"

def generate_intelligence(ai_percentage=30):
    """Generate a random boolean value for intelligence column
    
    Args:
        ai_percentage: Percentage chance of returning 'true' (0-100)
    
    Returns 'true' or 'false' as strings
    """
    # Convert percentage to probability (0.0 to 1.0)
    probability = ai_percentage / 100.0
    return "true" if random.random() < probability else "false"

def generate_template_type():
    """Generate a random template type or empty string
    
    Returns template type string or empty string (20% chance of empty for fallback testing)
    """
    # 20% chance of empty string to test fallback behavior
    if random.random() < 0.2:
        return ""
    
    return random.choice(TEMPLATE_TYPES)

def generate_test_csv(filename, num_entries=1000, ai_percentage=30):
    """Generate a CSV file with fake test data including enhanced features"""
    print(f"Generating {num_entries:,} fake entries with enhanced features...")
    print(f"AI Intelligence percentage: {ai_percentage}%")
    
    # Ensure uploads directory exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    filepath = uploads_dir / filename
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        # Updated fieldnames to include new columns
        fieldnames = ['name', 'company', 'linkedin_url', 'intelligence', 'template_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Generate entries
        progress_interval = max(1, num_entries // 20)  # Show progress 20 times
        
        for i in range(num_entries):
            name = generate_name()
            company = generate_company()
            linkedin_url = generate_linkedin_url(name)
            intelligence = generate_intelligence(ai_percentage)
            template_type = generate_template_type()
            
            writer.writerow({
                'name': name,
                'company': company,
                'linkedin_url': linkedin_url,
                'intelligence': intelligence,
                'template_type': template_type
            })
            
            # Progress indicator
            if (i + 1) % progress_interval == 0:
                progress = ((i + 1) / num_entries) * 100
                print(f"Progress: {i + 1:,}/{num_entries:,} entries ({progress:.1f}%)")
    
    file_size = filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Successfully generated {filepath}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"Total entries: {num_entries:,} + header = {num_entries + 1:,} lines")
    
    # Show statistics about generated data
    print("\nData Distribution:")
    intelligence_true_count = sum(1 for _ in range(num_entries) if random.random() < (ai_percentage / 100.0))
    template_empty_count = sum(1 for _ in range(num_entries) if random.random() < 0.2)
    print(f"  AI Intelligence (approx): ~{intelligence_true_count:,} entries (~{ai_percentage}%)")
    print(f"  Empty template_type (approx): ~{template_empty_count:,} entries (~20%)")
    print(f"  Available templates: {', '.join(TEMPLATE_TYPES)}")
    
    return str(filepath)

def generate_legacy_csv(filename, num_entries=1000):
    """Generate a CSV file with legacy format (backwards compatibility testing)"""
    print(f"Generating {num_entries:,} legacy format entries...")
    
    # Ensure uploads directory exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    filepath = uploads_dir / filename
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        # Legacy fieldnames (original format)
        fieldnames = ['name', 'company', 'linkedin_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Generate entries
        progress_interval = max(1, num_entries // 20)  # Show progress 20 times
        
        for i in range(num_entries):
            name = generate_name()
            company = generate_company()
            linkedin_url = generate_linkedin_url(name)
            
            writer.writerow({
                'name': name,
                'company': company,
                'linkedin_url': linkedin_url
            })
            
            # Progress indicator
            if (i + 1) % progress_interval == 0:
                progress = ((i + 1) / num_entries) * 100
                print(f"Progress: {i + 1:,}/{num_entries:,} entries ({progress:.1f}%)")
    
    file_size = filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Successfully generated {filepath} (legacy format)")
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"Total entries: {num_entries:,} + header = {num_entries + 1:,} lines")
    print("Note: This CSV uses the legacy format for backwards compatibility testing")
    
    return str(filepath)

def main():
    """Main function to handle command-line arguments"""
    # Default values
    default_entries = 1000
    default_ai_percentage = 30
    legacy_mode = False
    
    # Parse command-line arguments
    args = sys.argv[1:]
    ai_percentage = default_ai_percentage
    
    # Check for --legacy flag
    if '--legacy' in args:
        legacy_mode = True
        args.remove('--legacy')
    
    # Check for --ai-percent flag
    ai_percent_index = None
    for i, arg in enumerate(args):
        if arg.startswith('--ai-percent'):
            if '=' in arg:
                # Format: --ai-percent=50
                try:
                    ai_percentage = int(arg.split('=')[1])
                    args.remove(arg)
                except (ValueError, IndexError):
                    print("Error: --ai-percent must be followed by a valid integer (0-100)")
                    print_usage()
                    sys.exit(1)
            else:
                # Format: --ai-percent 50
                ai_percent_index = i
                break
    
    if ai_percent_index is not None:
        try:
            ai_percentage = int(args[ai_percent_index + 1])
            # Remove both the flag and its value
            args.pop(ai_percent_index + 1)  # Remove value first (higher index)
            args.pop(ai_percent_index)      # Then remove flag
        except (IndexError, ValueError):
            print("Error: --ai-percent must be followed by a valid integer (0-100)")
            print_usage()
            sys.exit(1)
    
    # Validate AI percentage
    if not 0 <= ai_percentage <= 100:
        print("Error: AI percentage must be between 0 and 100")
        sys.exit(1)
    
    # Parse remaining command-line arguments
    if len(args) == 0:
        # No arguments - use defaults
        num_entries = default_entries
        filename = f"test_data_{num_entries}.csv"
    elif len(args) == 1:
        # Only number of entries provided
        try:
            num_entries = int(args[0])
            filename = f"test_data_{num_entries}.csv"
        except ValueError:
            print("Error: Number of entries must be a valid integer")
            print_usage()
            sys.exit(1)
    elif len(args) == 2:
        # Both number and filename provided
        try:
            num_entries = int(args[0])
            filename = args[1]
            if not filename.endswith('.csv'):
                filename += '.csv'
        except ValueError:
            print("Error: Number of entries must be a valid integer")
            print_usage()
            sys.exit(1)
    else:
        print("Error: Too many arguments")
        print_usage()
        sys.exit(1)
    
    # Validate inputs
    if num_entries <= 0:
        print("Error: Number of entries must be positive")
        sys.exit(1)
    
    if num_entries > 1000000:
        print("Warning: Generating more than 1 million entries may take a long time and use significant disk space")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled")
            sys.exit(0)
    
    # Generate the data
    format_type = "legacy" if legacy_mode else "enhanced"
    print(f"Target: {num_entries:,} entries -> uploads/{filename} ({format_type} format)")
    
    if legacy_mode:
        generated_file = generate_legacy_csv(filename, num_entries)
    else:
        generated_file = generate_test_csv(filename, num_entries, ai_percentage)
    
    # Show recommended batch sizes for testing
    print("\nRecommended Batch Sizes for Testing:")
    for batch_size in [10, 25, 50, 100, 200]:
        batches = (num_entries + batch_size - 1) // batch_size  # Ceiling division
        print(f"   BATCH_SIZE={batch_size:3d} -> {batches:4d} batches")

def print_usage():
    """Print usage information"""
    print("""
Usage: python scripts/generate_test_data.py [number_of_entries] [output_filename] [--ai-percent N] [--legacy]

Arguments:
    number_of_entries    Number of CSV entries to generate (default: 1000)
    output_filename      Output filename (default: test_data_[number].csv)
    --ai-percent N       Percentage of AI intelligence rows (0-100, default: 30)
    --legacy            Generate legacy format CSV (name,company,linkedin_url only)

Examples:
    python scripts/generate_test_data.py                               # 1000 entries, 30% AI
    python scripts/generate_test_data.py 500                          # 500 entries, 30% AI
    python scripts/generate_test_data.py 2000 --ai-percent 50         # 2000 entries, 50% AI
    python scripts/generate_test_data.py 1000 test.csv --ai-percent=10 # 1000 entries, 10% AI
    python scripts/generate_test_data.py 5000 custom_data.csv         # 5000 entries, 30% AI
    python scripts/generate_test_data.py 1000 legacy_test.csv --legacy # 1000 entries (old format)

Enhanced format (default):
    name,company,linkedin_url,intelligence,template_type

Legacy format (--legacy):
    name,company,linkedin_url
    """)

if __name__ == "__main__":
    main()
