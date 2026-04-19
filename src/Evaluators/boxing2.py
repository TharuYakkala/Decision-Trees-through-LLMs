"""
boxing2 - 5 Zero-Shot Decision Tree Functions (Gemini 2.5 Flash)
"""

# Function 0 (from dt_function0.txt)
def dt_function_0(judge: str, official_judge: str, round_num: int):
    """
    Classifies the winner of a boxing round (Trinidad or de la Hoya) based on judge information and round number.

    Args:
        judge (str): The name of the judge (e.g., 'G. Hamada', 'HBO-Lederman').
        official_judge (str): 'yes' if the judge is official, 'no' otherwise.
        round_num (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing:
            - str: The predicted winner ('Trinidad' or 'de la Hoya').
            - list: A list of truth values (1 if condition satisfied, 0 otherwise)
                    for the inner nodes traversed to reach the prediction.
    """
    truth_values = [0, 0, 0]
    truth_values[0] = int(official_judge == 'yes')
    truth_values[1] = int(round_num <= 6)
    truth_values[2] = int(judge == 'HBO-Lederman')

    # Node 1 (Root): Check if the judge is official (most critical factor for a round's outcome).
    # Condition: official_judge == 'yes'
    if official_judge == 'yes':
        # Node 2 (Child of Node 1 'True' branch): For official judges, consider round dynamics.
        # Based on world knowledge for this specific fight (Trinidad vs. de la Hoya),
        # de la Hoya often started strong with boxing skill, while Trinidad was known for
        # late-fight aggression and power.
        # Condition: round_num <= 6 (early rounds)
        if round_num <= 6:
            return "de la Hoya", truth_values
        else: # round_num > 6 (later rounds)
            return "Trinidad", truth_values
    else: # official_judge == 'no'
        # Node 3 (Child of Node 1 'False' branch): For unofficial/media judges,
        # different outlets might have different leanings. HBO-Lederman was a prominent
        # analyst known for his technical scoring.
        # Condition: judge == 'HBO-Lederman'
        if judge == 'HBO-Lederman':
            return "de la Hoya", truth_values # Lederman often appreciated superior boxing technique.
        else: # Other unofficial judges/media (e.g., Sportsticker, LA Times, USA Today, AP)
            # Other media might lean towards the perceived aggression or narrative of Trinidad's power.
            return "Trinidad", truth_values



# Function 1 (from dt_function1.txt)

def dt_function_1(features):
    """
    Predicts the winner of a boxing round (Trinidad / de la Hoya) based on judging criteria.
    The decision tree has a maximum depth of 2.

    Parameters:
    features (dict): A dictionary containing the features for a given round.
                     Expected keys: 'Judge', 'Offical judge', 'Round'.

    Returns:
    tuple: A tuple containing:
           - str: The predicted winner ('Trinidad' or 'de la Hoya').
           - list: A list representing the truth values of the inner nodes.
                   [Node1_truth, Node2_truth, Node3_truth]
                   where:
                   - Node1_truth (index 0) corresponds to the condition 'Offical judge' == 'yes'.
                   - Node2_truth (index 1) corresponds to the condition 'Round' <= 6 (if Node1 is true).
                   - Node3_truth (index 2) corresponds to the condition 'Judge' == 'HBO-Lederman' (if Node1 is false).
                   An entry is 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise.
    """
    # Initialize truth values for all potential inner nodes.
    # The order is:
    # 0: Root node condition (Offical judge == 'yes')
    # 1: Left child node condition (Round <= 6) - relevant if root is true
    # 2: Right child node condition (Judge == 'HBO-Lederman') - relevant if root is false
    truth_values = [0, 0, 0]
    truth_values[0] = int(features['Offical judge'] == 'yes')
    truth_values[1] = int(features['Round'] <= 6)
    truth_values[2] = int(features['Judge'] == 'HBO-Lederman')
    # Node 1: Is it an official judge? This is the most crucial distinction.
    if features['Offical judge'] == 'yes':
        # Path for Official Judges: Often score based on sustained performance through the fight.
        # Hypothetically, de la Hoya might have started stronger with boxing, Trinidad came on later with power.
        if features['Round'] <= 6:
            return ("de la Hoya", truth_values) # Predict de la Hoya for early/mid-rounds by official judges
        else:
            return ("Trinidad", truth_values) # Predict Trinidad for later rounds by official judges
    else: # Offical judge == 'no' (i.e., a media or unofficial judge)
        # Path for Unofficial/Media Judges: Scoring can be more varied, and specific commentators
        # often have distinct styles or biases. HBO-Lederman was a prominent and often unique voice.
        if features['Judge'] == 'HBO-Lederman':
            return ("de la Hoya", truth_values) # HBO-Lederman often favored de la Hoya's boxing skills.
        else:
            return ("Trinidad", truth_values) # Other unofficial judges/media often saw Trinidad as winning or finishing stronger.



# Function 2 (from dt_function2.txt)
 
def dt_function_2(judge, official_judge, round_num):
    """
    Predicts the winner of a boxing round between Trinidad and de la Hoya
    based on judge type and round number.

    Args:
        judge (str): The name of the judge (e.g., 'G. Hamada', 'HBO-Lederman').
        official_judge (str): 'yes' if the judge is official, 'no' otherwise.
        round_num (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing:
            - str: The predicted winner ('Trinidad' or 'de la Hoya').
            - list: A list of integers (1 or 0) representing the truth values
                    of the inner node conditions encountered on the decision path.
                    1 if the condition is satisfied, 0 otherwise.
    """
    node_conditions = [0, 0]
    node_conditions[0] = int(official_judge == 'yes')
    node_conditions[1] = int(round_num <= 6)

    
    # Inner Node 1 (Root Node): Is the judge an official judge?
    # Condition: official_judge == 'yes'
    if official_judge == 'yes':
        # Path for Official Judges
        # Inner Node 2 (Child Node): Is it an early/mid round (1-6)?
        # Condition: round_num <= 6
        if round_num <= 6:
            prediction = "de la Hoya" # Often perceived as de la Hoya's strong early rounds by official judges
        else: # round_num > 6
            prediction = "Trinidad" # Often perceived as Trinidad's strong late rounds by official judges
    else: # official_judge == 'no'
        # Path for Unofficial Judges/Media
        # Inner Node 2 (Child Node): Is it an early/mid round (1-6)?
        # Condition: round_num <= 6
        if round_num <= 6:
            prediction = "de la Hoya" # Unofficial scores often favored de la Hoya's early boxing
        else: # round_num > 6
            prediction = "Trinidad" # Unofficial scores also recognized Trinidad's late surge

    return prediction, node_conditions


 
# Function 3 (from dt_function3.txt)
 
def dt_function_3(Judge, Official_judge, Round):
    """
    Induces a decision tree classifier to predict the winner of a boxing round
    between Trinidad and de la Hoya, based on profound world knowledge.

    Args:
        Judge (str): The name of the judge (e.g., 'G. Hamada', 'HBO-Lederman').
        Official_judge (str): 'yes' if the judge is an official boxing judge, 'no' otherwise.
        Round (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing:
            - str: The predicted winner of the round ('Trinidad' or 'de la Hoya').
            - list: A list representing the truth values of the inner nodes encountered.
                    An entry is 1 if the condition of the corresponding inner node
                    is satisfied, and 0 otherwise.
    """
    node_truth_values = [0, 0]
    node_truth_values[0] = int(Official_judge == 'yes')
    node_truth_values[1] = int(Round <= 6)

    # Node 1 (Root): Is the judge an official judge?
    # This is the most important distinction as official scores determine the outcome.
    if Official_judge == 'yes':        
        # Node 2 (Child of Node 1, for official judges): Is it an early round?
        # Based on general fight narratives, de la Hoya often started strong with his boxing skills,
        # while Trinidad was known for his relentless pressure and finishing strong in later rounds.
        if Round <= 6: # Considering the first half of the 12-round fight as 'early'
            return 'de la Hoya', node_truth_values
        else: # Round > 6 (later rounds)
            return 'Trinidad', node_truth_values
    else: # Official_judge == 'no' (e.g., media, commentators)
        # For unofficial judges, especially in a fight as controversial as Trinidad vs. de la Hoya,
        # many media outlets and observers felt de la Hoya had won many rounds due to his boxing acumen.
        # We'll assign de la Hoya as the default winner for rounds scored by unofficial sources.
        # This path leads to a leaf node at depth 1, so no further conditions are appended.
        return 'de la Hoya', node_truth_values


 
# Function 4 (from dt_function4.txt)
 
def dt_function_4(Judge, Official_judge, Round):
    """
    Predicts the winner of a boxing round (Trinidad vs. de la Hoya) based on judge information and round number.

    Args:
        Judge (str): The name of the judge/source (e.g., 'G. Hamada', 'HBO-Lederman').
        Official_judge (str): 'yes' if the judge is official, 'no' otherwise.
        Round (int): The round number (1 to 12).

    Returns:
        tuple: A tuple containing the predicted winner ('Trinidad' or 'de la Hoya')
               and a list of truth values for the inner nodes traversed.
               Each truth value is 1 if the condition is satisfied, 0 otherwise.
    """
    truth_values = [0, 0, 0]
    truth_values[0] = int(Official_judge == 'yes')
    truth_values[1] = int(Judge in ['G. Hamada', 'B. Logist'])
    truth_values[2] = int(Round <= 6)

    # Node 1 (Root): Is the judge an official judge?
    # Official judges' scores are primary determinants of the fight's outcome.
    if Official_judge == 'yes':

        # Node 2 (Depth 1, if Official Judge): Which official judge?
        # Based on historical results of the Trinidad vs. de la Hoya fight,
        # two of the three official judges (Hamada, Logist) scored for Trinidad,
        # while one (Roth) scored for de la Hoya.
        if Judge in ['G. Hamada', 'B. Logist']:
            prediction = 'Trinidad'
        else:  # Assuming the remaining official judge is J. Roth
            prediction = 'de la Hoya'
    else:  # Official_judge == 'no'

        # Node 3 (Depth 1, if Not Official Judge): What round is it?
        # Unofficial/media scores often reflect the overall momentum and narrative of the fight.
        # De la Hoya was widely perceived to win the earlier rounds due to his boxing skill and volume,
        # while Trinidad came on stronger in the later rounds with his power and aggression.
        if Round <= 6:
            prediction = 'de la Hoya'
        else:  # Round > 6 (later rounds)
            prediction = 'Trinidad'

    return prediction, truth_values

