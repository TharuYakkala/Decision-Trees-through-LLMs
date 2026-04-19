"""
colic - 5 Zero-Shot Decision Tree Functions (Gemini 2.5 Flash)
"""

  
# Function 0 (from dt_function0.txt)
  
def dt_function_0(Abdomen_Appearance: str, Degree_of_Pain: str) -> (str, list):
    """
    Induces a decision tree classifier to determine if a horse colic lesion is surgical or not.
    This function uses expert knowledge to identify the most critical features for
    a maximum depth of 2, without training on actual data.

    Args:
        Abdomen_Appearance (str): The appearance of the abdomen.
                                  Expected values: 'normal', 'other', 'feces in the large intestine',
                                  'distended small intestine', 'distended large intestine'.
        Degree_of_Pain (str): The observed degree of pain.
                              Expected values: 'none', 'mild', 'moderate', 'severe'.

    Returns:
        tuple: A tuple containing:
            - str: The predicted class ('surgical' or 'non-surgical').
            - list: A list representing the truth values of the inner nodes.
                    Each entry is 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise.
    """
    
    inner_node_truth_values = [0, 0]
    inner_node_truth_values[0] = int(Abdomen_Appearance in ['distended small intestine', 'distended large intestine'])
    inner_node_truth_values[1] = int(Degree_of_Pain == 'severe')

    prediction = None

    # Node 1: Abdomen Appearance
    # Condition: Is the abdomen appearance indicative of severe internal distension?
    if Abdomen_Appearance in ['distended small intestine', 'distended large intestine']:
        prediction = 'surgical'
    else:
        
        # Node 2 (Child of Node 1's FALSE branch): Degree of Pain
        # Condition: Is the horse experiencing severe pain?
        if Degree_of_Pain == 'severe':
            prediction = 'surgical'
        else:
            prediction = 'non-surgical'
            
    return prediction, inner_node_truth_values


  
# Function 1 (from dt_function1.txt)
  
def dt_function_1(features_dict):
    """
    Predicts whether a horse colic lesion requires surgery based on critical clinical features.
    
    Args:
        features_dict (dict): A dictionary containing the following features:
            - 'Abdominocentesis Appearance': (str) 'clear', 'cloudy', or 'serosanguinous'
            - 'Nasogastric Reflux': (str) 'none', '>1 liter', or '<1 liter'
            (Other features listed in the problem description are not used in this specific
            decision tree due to the maximum depth constraint and feature importance ranking.)
            
    Returns:
        tuple: A tuple containing:
            - str: 'yes' if surgery is predicted, 'no' otherwise.
            - list: A list representing the truth values of the inner nodes traversed.
                    1 if the condition was satisfied, 0 otherwise.
    """
    condition1 = features_dict['Abdominocentesis Appearance'] != 'clear'
    condition2 = features_dict['Nasogastric Reflux'] == '>1 liter'
    node_truth_values = [int(condition1), int(condition2)]

    # Node 1: Is Abdominocentesis Appearance abnormal (cloudy or serosanguinous)?
    # This is a strong indicator of peritonitis or compromised bowel.

    if condition1:
        # If abdominal fluid is not clear, it's highly indicative of a surgical lesion.
        prediction = 'yes'
    else:
        # If abdominal fluid is clear, evaluate the next most critical feature.
        # Node 2: Is there significant Nasogastric Reflux (>1 liter)?
        # Large volume reflux often indicates a small intestinal obstruction.

        if condition2:
            # Even with clear abdominal fluid, significant reflux points to surgery.
            prediction = 'yes'
        else:
            # If both abdominal fluid is clear and reflux is not significant,
            # the likelihood of requiring surgery is lower for this tree's depth.
            prediction = 'no'
            
    return prediction, node_truth_values


  
# Function 2 (from dt_function2.txt)
  
def dt_function_2(
    surgery, Age, Rectal_Temperature, Pulse, Respiratory_Rate, Temperature_of_Extremities,
    Strength_of_Peripheral_Pulse, Appearance_of_Mucous_Membranes, Capillary_Refill_Time,
    Degree_of_Pain, peristalsis, Abdominal_Distension, Nasogastric_Tube, Nasogastric_Reflux,
    Nasogastric_Reflux_pH, Rectal_Examination_Findings, Abdomen_Appearance, Packed_Cell_Volume,
    Total_Protein, Abdominocentesis_Appearance, Abdominocentesis_Total_Protein, Outcome
):
    node_truths = [0, 0, 0]
    node_truths[0] = int(Degree_of_Pain == 'severe')
    node_truths[0] = int(Nasogastric_Reflux == '>1 liter')
    node_truths[0] = int(Abdomen_Appearance in ('distended small intestine', 'distended large intestine'))


    # Node 1 (Root): Is the horse experiencing severe pain? Severe pain is a strong indicator for surgical intervention.
    if Degree_of_Pain == 'severe':

        # Node 2 (Left Child): If severe pain, is there also significant nasogastric reflux?
        # Large volumes of reflux often indicate a small intestinal obstruction requiring surgery.
        if Nasogastric_Reflux == '>1 liter':
            return "surgical", node_truths
        else:
            # Even without significant reflux, severe and intractable pain usually necessitates surgical exploration.
            return "surgical", node_truths
    else:

        # Node 2 (Right Child): If pain is not severe, are there clear signs of intestinal distension?
        # Distended small or large intestine strongly suggests an obstruction or displacement.
        if Abdomen_Appearance in ('distended small intestine', 'distended large intestine'):
            # Even with moderate pain, clear physical signs of obstruction usually lead to surgery.
            return "surgical", node_truths
        else:
            # Without severe pain or obvious intestinal distension, the case is more likely medically manageable.
            return "not surgical", node_truths


  
# Function 3 (from dt_function3.txt)
  
def dt_function_3(features):
    """
    Predicts whether a horse colic lesion is surgical or not based on key clinical features.

    Parameters:
    -----------
    features : dict
        A dictionary containing the horse's clinical features. Expected keys:
        - 'Degree of Pain' (str): 'none', 'mild', 'moderate', 'severe'
        - 'Abdomen Appearance' (str): 'normal', 'other', 'feces in the large intestine',
                                     'distended small intestine', 'distended large intestine'

    Returns:
    --------
    tuple:
        - prediction (str): 'surgical' or 'not surgical'
        - nodes_truth_values (list): A list of integers (0 or 1) representing
                                     the truth values of the inner node conditions.
                                     The first element corresponds to the root node condition,
                                     and the second element corresponds to the condition of the
                                     inner node in the "not severe pain" branch.
    """
    cond1_satisfied = (features['Degree of Pain'] == 'severe')
    cond2_satisfied = (features['Abdomen Appearance'] in ('distended small intestine', 'distended large intestine'))

    nodes_truth_values = [int(cond1_satisfied), int(cond2_satisfied)]
    prediction = None

    # Inner Node 1 (Root Node Condition): Is the degree of pain severe?
    # Rationale: Severe, intractable pain is the single most urgent indicator of a surgical colic.

    if cond1_satisfied:
        # If pain is severe, it's highly indicative of a surgical lesion.
        # This path leads directly to a prediction as it's a critical, often immediate, indicator.
        prediction = 'surgical'
        # The second inner node's condition is not evaluated on this path. As per example's
        # interpretation, we append 0 for the condition not reached/satisfied on this specific path.
    else:
        # If pain is not severe, we proceed to evaluate other significant physical findings.

        # Inner Node 2 (Condition on the 'not severe pain' branch):
        # Is there significant abdominal distension (suggesting obstruction or displacement)?
        # Rationale: Even without severe pain, distended intestines on examination strongly point
        #            to an obstructive or displacement lesion usually requiring surgical intervention.
        if cond2_satisfied:
            # If significant distension is present, it indicates a likely surgical case.
            prediction = 'surgical'
        else:
            # If pain is not severe AND no significant abdominal distension,
            # the likelihood of a surgical lesion is considerably lower,
            # leaning towards a medical colic or a mild, resolving issue.
            prediction = 'not surgical'

    return prediction, nodes_truth_values


  
# Function 4 (from dt_function4.txt)
  
def dt_function_4(
    surgery, Age, Rectal_Temperature, Pulse, Respiratory_Rate,
    Temperature_of_Extremities, Strength_of_Peripheral_Pulse,
    Appearance_of_Mucous_Membranes, Capillary_Refill_Time,
    Degree_of_Pain, peristalsis, Abdominal_Distension, Nasogastric_Tube,
    Nasogastric_Reflux, Nasogastric_Reflux_pH, Rectal_Examination_Findings,
    Abdomen_Appearance, Packed_Cell_Volume, Total_Protein,
    Abdominocentesis_Appearance, Abdominocentesis_Total_Protein,
    Outcome
):
    """
    Classifies whether a horse colic lesion is surgical or not, based on key features.
    The decision tree has a maximum depth of 2.

    Args:
        All features listed in the problem description, as individual arguments.

    Returns:
        A tuple containing:
        - A string: "surgical" or "not surgical"
        - A list of integers: Representing the truth values (1 for true, 0 for false)
          of the conditions encountered at each inner node along the decision path.
    """
    node_truths = [0, 0]
    node_truths[0] = int(Degree_of_Pain == 'severe')
    node_truths[1] = int(Nasogastric_Reflux == '>1 liter')

    # Node 1 (Root Node): Is the degree of pain severe?
    # This is a very strong indicator for surgical intervention.
    if Degree_of_Pain == 'severe':
        # If pain is severe, it's highly indicative of a surgical case.
        return "surgical", node_truths
    else:

        # Node 2 (Child Node): Is there significant nasogastric reflux (>1 liter)?
        # Significant reflux often indicates a functional or physical obstruction
        # that may require surgery, even if pain is not yet severe.
        if Nasogastric_Reflux == '>1 liter':
            return "surgical", node_truths
        else:
            # If not severe pain and no significant reflux, leans towards non-surgical management.
            return "not surgical", node_truths

