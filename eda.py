from collections import Counter
from ast import literal_eval
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns


def terms_statistics(df):
    # Ensure each item in 'terms' is correctly formatted as a list
    df['terms'] = df['terms'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    
    # Split terms by semicolon and comma, and flatten the list
    def split_and_flatten(terms):
        # Split by semicolon and comma, then flatten
        flattened_terms = []
        for term in terms:
            split_terms = sum([t.split('; ') for t in term.split(', ')], [])
            flattened_terms.extend(split_terms)
        return flattened_terms
    
    df['terms'] = df['terms'].apply(split_and_flatten)
    
    # Explode the 'terms' column to separate rows
    exploded_terms = df.explode('terms')
    
    # Trim whitespace and remove empty strings
    exploded_terms['terms'] = exploded_terms['terms'].str.strip().replace('', None)
    
    # Count occurrences of each term
    label_counts = Counter(exploded_terms['terms'].dropna())
    num_unique_terms = len(label_counts)
    print(f"Number of unique terms: {num_unique_terms}")

    terms_appearing_once = {term: count for term, count in label_counts.items() if count == 1}
    num_terms_appearing_once = len(terms_appearing_once)

    print(f"Total number of terms that appear only once: {num_terms_appearing_once}")
    print("Most common labels:", label_counts.most_common(10))

    # Keep rows where the same term appears more than once across all entries
    filtered_combined_arxiv = df.explode('terms').groupby('terms').filter(lambda x: len(x) > 1)

    # Now, count occurrences of each term in the filtered DataFrame
    filtered_label_counts = Counter(filtered_combined_arxiv['terms'])
    num_unique_terms_filtered = len(filtered_label_counts)
    print(f"Number of unique terms after filtering: {num_unique_terms_filtered}")
    filtered_term_freq_df = pd.DataFrame(filtered_label_counts.items(), columns=['Term', 'Frequency']).sort_values(by='Frequency', ascending=False)
    
    # Group by 'titles' and 'abstracts', and aggregate 'terms' into a list
    processed_filtered_combined_arxiv = filtered_combined_arxiv.groupby(['titles', 'abstracts'])['terms'].apply(list).reset_index()

    # Ensure each list of terms is unique
    processed_filtered_combined_arxiv['terms'] = processed_filtered_combined_arxiv['terms'].apply(lambda x: list(set(x)))

    return filtered_combined_arxiv, filtered_term_freq_df, processed_filtered_combined_arxiv



def dashboard_terms(filtered_term_freq_df, processed_filtered_combined_arxiv):
    # Setup the figure and axes for a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    # Plot 1: Top 40 Most Frequent Terms
    top_20_terms = filtered_term_freq_df.head(20)  # Adjusting to get top 40
    sns.barplot(x='Frequency', y='Term', data=top_20_terms, palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Top 40 Most Frequent Terms')
    axes[0, 0].set_xlabel('Count of Terms')
    axes[0, 0].set_ylabel('Term Labels')

    # Plot 2: Venn Diagram of Top 3 Labels
    top_3_labels = filtered_term_freq_df.head(3)['Term'].values
    sets = {label: set(processed_filtered_combined_arxiv[processed_filtered_combined_arxiv['terms'].apply(lambda x: label in x)].index) for label in top_3_labels}
    venn_sets = {
        '100': len(sets[top_3_labels[0]] - sets[top_3_labels[1]] - sets[top_3_labels[2]]),
        '010': len(sets[top_3_labels[1]] - sets[top_3_labels[0]] - sets[top_3_labels[2]]),
        '001': len(sets[top_3_labels[2]] - sets[top_3_labels[0]] - sets[top_3_labels[1]]),
        '110': len(sets[top_3_labels[0]] & sets[top_3_labels[1]] - sets[top_3_labels[2]]),
        '101': len(sets[top_3_labels[0]] & sets[top_3_labels[2]] - sets[top_3_labels[1]]),
        '011': len(sets[top_3_labels[1]] & sets[top_3_labels[2]] - sets[top_3_labels[0]]),
        '111': len(sets[top_3_labels[0]] & sets[top_3_labels[1]] & sets[top_3_labels[2]]),
    }
    venn3(subsets=venn_sets, set_labels=top_3_labels, ax=axes[0, 1])
    axes[0, 1].set_title('Venn Diagram of Top 3 Labels')

    # Plot 3: Top 20 Primary Categories for 'cs.CV'
    cs_CV_papers = processed_filtered_combined_arxiv[processed_filtered_combined_arxiv['terms'].apply(lambda x: 'cs.CV' in x)]
    all_categories = [term for sublist in cs_CV_papers['terms'] for term in sublist if term != 'cs.CV']
    category_counts = Counter(all_categories)
    top_20_categories = pd.DataFrame(category_counts.most_common(20), columns=['Category', 'Count']).sort_values(by='Count', ascending=True)
    sns.barplot(x='Count', y='Category', data=top_20_categories, orient='h', palette='Blues_d', ax=axes[1, 0])
    # Add value annotations to each bar
    # for index, (cat, count) in enumerate(zip(top_20_categories['Category'], top_20_categories['Count'])):
    #     plt.text(count, index, f" {count}", va='center', fontsize=10)
    axes[1, 0].set_title('Top 20 Primary Categories associated with "cs.CV" Papers')
    axes[1, 0].set_xlabel('Paper Count')
    axes[1, 0].set_ylabel('Primary Category')

    # Invert y-axis to have the highest count at the top
    axes[1, 0].invert_yaxis()

    # Add value annotations to each bar
    for index, (cat, count) in enumerate(zip(top_20_categories['Category'], top_20_categories['Count'])):
        axes[1, 0].text(count, index, f" {count}", va='center', ha='left', color='white')

  
    # Plot 4: Pareto Plot of Primary Categories
    # Step 1: Flatten the 'terms' lists and count occurrences of all categories
    all_categories = [term for sublist in processed_filtered_combined_arxiv['terms'] for term in sublist]
    category_counts = Counter(all_categories)
    # Step 2: Convert the counter to a DataFrame
    category_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])
    # Step 3: Sort the DataFrame to get the top categories based on count
    category_df.sort_values(by='Count', ascending=False, inplace=True)
    # Calculate the total counts for percentage calculation
    total_count = category_df['Count'].sum()
    # Calculate the cumulative percentage for all categories
    category_df['Cumulative_Percentage'] = (category_df['Count'].cumsum() / total_count * 100)

    # Step 4: Filter the DataFrame to get the top 20 categories for the bar plot
    top_20_categories = category_df.head(20)
    # Create the bar plot with categories on the x-axis
    sns.barplot(x='Category', y='Count', data=top_20_categories, color='skyblue', ax=axes[1, 1])
    axes[1, 1].tick_params(axis='x', rotation=90)  # Rotate category labels for better readability
    axes[1, 1].set_title('Pareto Plot of Primary Categories')
    axes[1, 1].set_xlabel('Primary Category')
    axes[1, 1].set_ylabel('Paper Count')
    # Create a secondary y-axis for the cumulative percentage
    sec_ax = axes[1, 1].twinx()
    # Plot the cumulative percentage line using the full category_df
    sec_ax.plot(category_df['Category'].head(20), category_df['Cumulative_Percentage'].head(20), color='red', marker='o', linestyle='-', label='Cumulative Percentage')
    # Annotate the cumulative percentage on the line
    for i, (pct, cat) in enumerate(zip(category_df['Cumulative_Percentage'].head(20), category_df['Category'].head(20))):
        if i % 2 == 0:
            sec_ax.text(i, pct, f"{pct:.2f}%", fontsize=10, color='white', va='bottom', ha='center')
    # Sync the x-axis limits for the secondary axis
    sec_ax.set_xlim(axes[1, 1].get_xlim())
    sec_ax.set_ylim(0, 110)  # Adjust the percentage limits if necessary
    sec_ax.set_ylabel('Cumulative Percentage (%)')
    # Add a legend for the cumulative percentage line
    sec_ax.legend(loc='upper left')
    sec_ax.grid(False)


from sklearn.preprocessing import MultiLabelBinarizer

def multi_bin(df):
    # Flatten the list of lists in 'terms' column to get all labels in a single list
    all_labels = [label for sublist in df['terms'] for label in sublist]
    
    # # Count the frequency of each unique label and sort them by frequency in descending order
    # label_counts = Counter(all_labels)
    # sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    # print("Frequency of each label in 'terms' (sorted by frequency):")
    # for label, count in sorted_label_counts:
    #     print(f"{label}: {count}")
    
    # Extract unique labels (optional, as MultiLabelBinarizer does this internally)
    unique_labels = set(all_labels)
    print(f"\nUnique labels in 'terms': {unique_labels}")
    print(f"\nNumber of unique labels: {len(unique_labels)}")
    
    mlb = MultiLabelBinarizer()
    encoded_terms = mlb.fit_transform(df['terms'])
    terms_df = pd.DataFrame(encoded_terms, columns=mlb.classes_)
    
    df_ready_encoded = pd.concat([df.reset_index(drop=True), terms_df.reset_index(drop=True)], axis=1)
    
    print("\n", df_ready_encoded.head())
    print("Number of unique columns (including original ones):", df_ready_encoded.shape[1])

    return df_ready_encoded