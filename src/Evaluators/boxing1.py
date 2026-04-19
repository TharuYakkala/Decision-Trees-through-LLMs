"""
boxing1 - 5 Zero-Shot Decision Tree Functions (Gemini 2.5 Flash)
"""

# Function 0 (from dt_function0.txt)
def dt_function_0(Judge, Official_judge, Round):
    """
    Induces a decision tree classifier to predict the winner of a boxing round
    between Lewis and Holyfield, based on "world knowledge" without real-world data.

    The tree classifies the winner of a specific round based on who was scoring
    (an official judge or media) and, if official, which judge it was.
    This decision tree is built with a maximum depth of 2.

    Args:
        Judge (str): The name of the judge (e.g., 'E. Williams', 'HBO-Lederman').
                     Possible values: 'E. Williams', 'L. O\'Connell', 'S. Christodoulu',
                     'HBO-Lederman', 'Sportsticker', 'Boxing Times', 'Sportsline',
                     'Associated Press', 'ESPN', 'Boxing Monthly-Leach'.
        Official_judge (str): 'yes' if the judge is an official fight judge, 'no' otherwise.
        Round (int): The round number (1 to 12). This feature is not used in this
                     depth-2 tree due to higher importance of other features.

    Returns:
        tuple:
            - str: The predicted winner of the round ('Lewis' or 'Holyfield').
            - list: A list representing the truth values of the inner nodes traversed.
                    Each entry is 1 if the condition of the corresponding inner node
                    is satisfied, and 0 otherwise.
                    The list reflects the specific path taken through the decision tree.
    """
    inner_node_truth_values = [0, 0]
    inner_node_truth_values[0] = int(Official_judge == 'yes')
    inner_node_truth_values[1] = int(Judge == 'E. Williams')
    
    # Root Node (Depth 1): Is the scoring from an official judge?
    # Based on world knowledge, official judges' scores directly determine the fight outcome
    # and are subject to different dynamics than media scorecards.
    # Condition 1: Official_judge == 'yes'
    if Official_judge == 'yes':        
        # Second Level Node (Depth 2, Left Branch): Which specific official judge?
        # Based on world knowledge of the controversial Lewis-Holyfield I fight,
        # Judge E. Williams was notable for scoring the fight for Holyfield.
        # Condition 2: Judge == 'E. Williams'
        if Judge == 'E. Williams':
            prediction = 'Holyfield'
        else:
            # For other official judges (e.g., L. O'Connell, S. Christodoulu, or others),
            # many considered Lewis to have won more rounds or the fight overall.
            prediction = 'Lewis'
            
    else: # Official_judge == 'no'        
        # For unofficial scorecards (media, pundits), the prevailing sentiment
        # after Lewis-Holyfield I was that Lewis had won the fight.
        prediction = 'Lewis'
        
        # No further internal node is evaluated on this path as max depth is 2,
        # and this branch terminates after the root node decision.
    
    return prediction, inner_node_truth_values


# Function 1 (from dt_function1.txt)
def dt_function_1(judge, official_judge, round_num):
    """
    Predicts the winner of a boxing round between Lewis and Holyfield based on
    judge information, reflecting common perceptions of their fights, especially
    the first controversial bout.

    Args:
        judge (str): The name of the judge (e.g., 'E. Williams', 'HBO-Lederman').
        official_judge (str): 'yes' if the judge is official, 'no' otherwise.
        round_num (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing:
            - str: The predicted winner ('Lewis' or 'Holyfield').
            - list: A list representing the truth values of the inner nodes
                    traversed. 1 if the condition was satisfied, 0 otherwise.
    """
    node_truth_values = [0, 0]
    node_truth_values[0] = int(official_judge == 'yes')
    node_truth_values[1] = int(judge == 'S. Christodoulu')

    # Node 0: Is the judge an official judge?
    if official_judge == 'yes':
        # Node 1 (nested under Node 0 true branch): Is the official judge S. Christodoulu?
        if judge == 'S. Christodoulu':
            return "Lewis", node_truth_values # S. Christodoulu famously scored for Lewis
        else: # Implies E. Williams or L. O'Connell
            return "Holyfield", node_truth_values # These judges were key to the draw or scored for Holyfield
    else:  # official_judge == 'no'
        # For non-official judges (media), Lewis was widely perceived as the winner
        return "Lewis", node_truth_values



# Function 2 (from dt_function2.txt)
def dt_function_2(Judge: str, Offical_judge: str, Round: int) -> tuple[str, list[int]]:
    """
    Predicts the winner of a boxing round between Lewis and Holyfield based on
    judge type, specific judge, and round number.

    Args:
        Judge (str): The name of the judge (e.g., 'E. Williams', 'HBO-Lederman', 'Sportsticker').
        Offical_judge (str): 'yes' if the score is from an official judge, 'no' otherwise.
        Round (int): The round number (1 to 12).

    Returns:
        tuple[str, list[int]]: A tuple containing the predicted winner ('Lewis' or 'Holyfield')
                               and a list of truth values for inner nodes.
                               Truth values are 1 if the condition of the corresponding inner node
                               is satisfied, and 0 otherwise.
    """
    node_truth_values = [0, 0, 0]
    node_truth_values[0] = int(Offical_judge == 'yes')
    node_truth_values[1] = int(Round <= 6)
    node_truth_values[2] = int(Judge == 'HBO-Lederman')
    
    # Root node: Is it an official judge's score?
    # Condition 1: Offical_judge == 'yes'
    if Offical_judge == 'yes':
        # First-level node (Path 1): If official, consider the round number
        # Condition 2: Round <= 6 (Early rounds vs. Late rounds)
        if Round <= 6:
            # Prediction for official, early rounds: Lewis (often dominant early with cleaner boxing)
            return "Lewis", node_truth_values
        else:
            # Prediction for official, late rounds: Holyfield (known for stamina, strong finish)
            return "Holyfield", node_truth_values
    else:
        # First-level node (Path 2): If unofficial, consider specific media/judge biases
        # Condition 3: Judge == 'HBO-Lederman'
        if Judge == 'HBO-Lederman':
            # Prediction for HBO-Lederman: Lewis (known for scoring Lewis heavily in their controversial fights)
            return "Lewis", node_truth_values
        else:
            # Prediction for other unofficial judges: Lewis (general media perception often favored Lewis's boxing skill)
            return "Lewis", node_truth_values


# Function 3 (from dt_function3.txt)
def dt_function_3(Judge, Offical_judge, Round):
    """
    Classifies the winner of a boxing round between Lewis and Holyfield based on judge information.

    Args:
        Judge (str): The name of the scoring judge/source (e.g., 'E. Williams', 'HBO-Lederman').
        Offical_judge (str): 'yes' if the judge is official, 'no' otherwise.
        Round (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing:
            - str: The predicted winner ('Lewis' or 'Holyfield').
            - list: A list representing the truth values of the inner nodes traversed.
                    An entry is 1 if the node's condition is satisfied, 0 otherwise.
    """
    inner_node_truth_values = [0, 0]
    inner_node_truth_values[0] = int(Offical_judge == 'yes')
    inner_node_truth_values[1] = int(Judge == 'E. Williams')
    # Node 1 (Root): Is the scorer an official judge?
    # Condition: Offical_judge == 'yes'
    if Offical_judge == 'yes':
        # Path for official judges (e.g., E. Williams, L. O'Connell, S. Christodoulu)

        # Node 2 (Child of Node 1's True branch): Is the official judge E. Williams?
        # Based on world knowledge of the controversial Lewis-Holyfield I fight,
        # Eugenia Williams famously scored heavily for Holyfield.
        # Condition: Judge == 'E. Williams'
        if Judge == 'E. Williams':
            prediction = 'Holyfield'
        else:
            # For other official judges, Lewis is generally perceived to have won more rounds
            # or was the more dominant boxer overall, especially in neutral scoring.
            prediction = 'Lewis'
    else:
        # Path for unofficial judges / media outlets (e.g., HBO-Lederman, ESPN)
        # Media and public consensus often favored Lewis in their controversial bouts.
        prediction = 'Lewis'

    return prediction, inner_node_truth_values



# Function 4 (from dt_function4.txt)
def dt_function_4(Judge, Official_judge, Round):
    """
    Classifies the winner of a boxing round between Lewis and Holyfield based on
    judge type and round number, reflecting typical boxing perceptions.

    Args:
        Judge (str): The name of the judge (e.g., 'E. Williams', 'HBO-Lederman').
        Official_judge (str): 'yes' if the judge is official, 'no' otherwise.
        Round (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing the predicted winner ('Lewis' or 'Holyfield')
               and a list of truth values for the inner nodes.
               Truth values are 1 if the condition is satisfied, 0 otherwise.
               The list can have 1 or 2 elements based on the tree depth reached.
    """
    node_truth_values = [0, 0]
    node_truth_values[0] = int(Official_judge == 'yes')
    node_truth_values[1] = int(Round <= 6)
    # Node 1: Is this an official judge? (Most critical distinction)
    # Condition: Official_judge == 'yes'
    if Official_judge == 'yes':
        
        # Node 2 (Depth 2): For official judges, early vs. late rounds often show different patterns.
        # Holyfield was known for his early aggression, Lewis for sustained boxing and power in later rounds.
        # Condition: Round <= 6 (Early rounds)
        if Round <= 6:
            return "Holyfield", node_truth_values
        else:  # Round > 6 (Later rounds)
            return "Lewis", node_truth_values
    else:  # Official_judge == 'no' (Unofficial sources)
        
        # Leaf Node (Depth 1): Unofficial sources, especially after their controversial first fight,
        # often leaned towards Lewis's more dominant boxing performance.
        return "Lewis", node_truth_values

