import pandas as pd

def analyze_traj(file_path):
    df = pd.read_csv(file_path)
    group_size = 15
    improvements = []

    for start in range(0, len(df), group_size):
        group = df.iloc[start:start+group_size]
        scores = group['score'].dropna().astype(str)
        distinct_scores = scores.unique()
        improvements.append(len(distinct_scores))
    
    for idx, count in enumerate(improvements, 1):
        print(f'Group {idx}: {count} improvements')
    print()
    print(f'Average improvements per group: {sum(improvements)/len(improvements)}')
    print()
    print(f'Total improvements: {sum(improvements)}')
    
    # Calculate average score for each position in the group
    position_scores = [[] for _ in range(group_size)]
    
    for start in range(0, len(df), group_size):
        group = df.iloc[start:start+group_size]
        for i, score in enumerate(group['score'].dropna()):
            if i < group_size:
                try:
                    position_scores[i].append(float(score))
                except ValueError:
                    pass  # Ignore non-numeric scores

    for i, scores in enumerate(position_scores, 1):
        if scores:
            average = sum(scores) / len(scores)
            print(f'Average score for position {i}: {average}')
        else:
            print(f'Average score for position {i}: 0')
            
    import matplotlib.pyplot as plt

    # Compute average score per position
    average_position_scores = [sum(scores)/len(scores) if scores else 0 for scores in position_scores]

    # Compute average over 5 subgroups of 3 positions
    subgroup_averages = []
    for i in range(0, group_size, 3):
        subgroup = position_scores[i:i+3]
        avg = sum([sum(scores)/len(scores) if scores else 0 for scores in subgroup]) / 3
        subgroup_averages.append(avg)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Line graph for average_position_scores
    positions = range(1, group_size + 1)
    ax1.plot(positions, average_position_scores, color='b', marker='o', label='Average Score per Position')
    ax1.set_xlabel('Position in Group')
    ax1.set_ylabel('Average Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Bar graph for subgroup_averages using the same y-scale
    subgroup_labels = [f'Group {i+1}' for i in range(len(subgroup_averages))]
    bar_positions = [i * 3 + 2 for i in range(len(subgroup_averages))]
    ax1.bar(bar_positions, subgroup_averages, width=2, alpha=0.5, color='g', label='Average Score per Subgroup')
    
    # Title and legends
    plt.title('Average Scores per Position and per Subgroup')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # analyze_traj('Compfiles/final/traj.csv')
    analyze_traj('final_traj.csv')