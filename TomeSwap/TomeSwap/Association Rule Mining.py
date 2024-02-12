# Association Rule Mining Algorithm (Example: mlxtend library)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Sample transactional dataset (replace 'transactions.csv' with your dataset)
transactions = [
    ['book1', 'book2', 'book3'],
    ['book2', 'book4'],
    ['book1', 'book2', 'book4'],
    ['book1', 'book3'],
    ['book2', 'book3', 'book4']
]

# Convert transactions to a one-hot encoded DataFrame
onehot_df = pd.get_dummies(pd.DataFrame(transactions), prefix='', prefix_sep='')

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(onehot_df, min_support=0.5, use_colnames=True)

# Generate association rules from frequent itemsets
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Print association rules
print("Association Rules:")
print(rules)
