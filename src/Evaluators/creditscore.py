"""
creditscore - 5 Zero-Shot Decision Tree Functions (Gemini 2.5 Flash)
"""

   
# Function 0 (from dt_function0.txt)
   
def dt_function_0(age, income_per_dependent, monthly_credit_card_expenses, owning_a_home, self_employed, number_of_derogatory_reports):
    """
    Classifies a credit application as 'Accepted' or 'Rejected' based on applicant features.

    Args:
        age (int): Applicant's age in years.
        income_per_dependent (float): Applicant's income per dependent (1.5 to 10).
        monthly_credit_card_expenses (float): Applicant's monthly credit card expenses in dollars.
        owning_a_home (str): 'yes' if applicant owns a home, 'no' otherwise.
        self_employed (str): 'yes' if applicant is self-employed, 'no' otherwise.
        number_of_derogatory_reports (int): Number of derogatory reports on the applicant's credit history.

    Returns:
        tuple: A tuple containing:
            - str: The prediction ('Accepted' or 'Rejected').
            - list: A list of integers (0 or 1) representing the truth values of the inner nodes encountered.
                    1 if the node's condition is satisfied, 0 otherwise.
    """
    inner_node_truth_values = [0, 0]
    inner_node_truth_values[0] = int(number_of_derogatory_reports <= 0)
    inner_node_truth_values[1] = int(income_per_dependent > 4.0)


    # Node 1: Check the number of derogatory reports (a primary indicator of credit risk)
    # Condition: number_of_derogatory_reports <= 0 (Applicant has no derogatory reports)
    if number_of_derogatory_reports <= 0:

        # Node 2 (child of Node 1 True branch): If no derogatory reports, check income per dependent
        # Condition: income_per_dependent > 4.0 (A higher income per dependent generally indicates better repayment capacity)
        if income_per_dependent > 4.0:
            prediction = "Accepted"
        else:
            prediction = "Rejected"  # Not enough income despite clean record
    else:
        prediction = "Rejected"  # Presence of derogatory reports typically leads to rejection

    return prediction, inner_node_truth_values


   
# Function 1 (from dt_function1.txt)
   
def dt_function_1(features):
    """
    Predicts whether a credit application is accepted or rejected based on provided features.

    Args:
        features (dict): A dictionary containing the applicant's features.
                         Expected keys:
                         - 'age (years)'
                         - 'income per dependent (1.5 to 10)'
                         - 'monthly credit card expenses ($)'
                         - 'owning a home (yes / no)'
                         - 'self employed (yes / no)'
                         - 'number of derogatory reports'

    Returns:
        tuple: A tuple containing:
               - str: The prediction ('Accepted' or 'Rejected').
               - list: A list of truth values for the inner nodes.
                       [1st_node_condition_met, 2nd_node_condition_met]
                       1 if the condition is satisfied, 0 otherwise.
    """
    node_truths = [0, 0]  # Initialize truth values for the two potential inner nodes

    # Extract relevant features using provided names
    number_of_derogatory_reports = features['number of derogatory reports']
    income_per_dependent = features['income per dependent (1.5 to 10)']

    # --- Decision Tree Logic (Max Depth 2) ---
    node_truths[0] = int(number_of_derogatory_reports > 0)
    node_truths[1] = int(income_per_dependent >= 4.0)
    # Node 1 (Root Node): Check for severe negative indicators - derogatory reports
    if number_of_derogatory_reports > 0:
        prediction = "Rejected"
    else:
        
        # Node 2 (Child of Node 1's 'False' branch): Evaluate repayment capacity
        # This node is only reached if there are no derogatory reports.
        if income_per_dependent >= 4.0:  # A reasonable threshold for sufficient income per dependent
            prediction = "Accepted"
        else:
            prediction = "Rejected"
            
    return prediction, node_truths


   
# Function 2 (from dt_function2.txt)
   
def dt_function_2(age, income_per_dependent, monthly_credit_card_expenses, owning_a_home, self_employed, number_of_derogatory_reports):
    """
    Classifies a credit application as 'Accepted' or 'Rejected' based on a decision tree
    with a maximum depth of 2.

    Args:
        age (int): Applicant's age in years.
        income_per_dependent (float): Applicant's income per dependent (1.5 to 10 scale).
        monthly_credit_card_expenses (float): Applicant's monthly credit card expenses in dollars.
        owning_a_home (str): 'yes' if applicant owns a home, 'no' otherwise.
        self_employed (str): 'yes' if applicant is self-employed, 'no' otherwise.
        number_of_derogatory_reports (int): Number of derogatory reports on applicant's credit history.

    Returns:
        tuple: A tuple containing:
            - str: The prediction ('Accepted' or 'Rejected').
            - list: A list of integers representing the truth values of the inner nodes.
                    The list corresponds to [Node1, Node2, Node3] where:
                    Node1: (number_of_derogatory_reports == 0)
                    Node2: (income_per_dependent >= 3.0) - evaluated if Node1 is True
                    Node3: (income_per_dependent >= 7.0) - evaluated if Node1 is False
                    1 if the condition of the corresponding inner node is satisfied, 0 otherwise.
    """
    
    # Initialize truth values for the three potential inner nodes
    node1_satisfied = int(number_of_derogatory_reports == 0) # Root node: number_of_derogatory_reports == 0
    node2_satisfied = int(income_per_dependent >= 3.0)# Left child of root: income_per_dependent >= 3.0
    node3_satisfied = int(income_per_dependent >= 7.0) # Right child of root: income_per_dependent >= 7.0
    
    final_nodes = [node1_satisfied, node2_satisfied, node3_satisfied]
    # Root Node Condition: number_of_derogatory_reports == 0
    if number_of_derogatory_reports == 0:
        # Path: No derogatory reports
        
        # Second Level Node (Left Branch): income_per_dependent >= 3.0
        if income_per_dependent >= 3.0:
            prediction = "Accepted"
        else:
            prediction = "Rejected" # Low income despite clean record
            
    else: # number_of_derogatory_reports > 0
        # Path: One or more derogatory reports
        
        # Second Level Node (Right Branch): income_per_dependent >= 7.0
        if income_per_dependent >= 7.0:
            prediction = "Accepted" # High income offsets derogatory reports
        else:
            prediction = "Rejected" # Derogatory reports and not exceptionally high income
            
    return prediction, final_nodes


   
# Function 3 (from dt_function3.txt)
   
def dt_function_3(age, income_per_dependent, monthly_credit_card_expenses, owning_a_home, self_employed, number_of_derogatory_reports):
    """
    Classifies whether a credit application is accepted or not based on
    a decision tree induced from world knowledge.

    Args:
        age (int): Applicant's age in years.
        income_per_dependent (float): Income per dependent, ranging from 1.5 to 10.
        monthly_credit_card_expenses (float): Monthly credit card expenses in dollars.
        owning_a_home (str): 'yes' or 'no' indicating home ownership.
        self_employed (str): 'yes' or 'no' indicating self-employment status.
        number_of_derogatory_reports (int): Number of derogatory reports on credit history.

    Returns:
        tuple: (prediction, node_truth_values)
               prediction (str): 'Accepted' or 'Denied'.
               node_truth_values (list): List of 1s and 0s representing truth values
                                         of inner nodes. 1 if condition satisfied, 0 otherwise.
    """
    node_truth_values = [0, 0]
    node_truth_values[0] = int(number_of_derogatory_reports > 0)
    node_truth_values[1] = int(income_per_dependent >= 5.0)

    # Node 1: Check for derogatory reports (most critical factor for credit denial)
    # Condition: number_of_derogatory_reports > 0
    if number_of_derogatory_reports > 0:
        prediction = 'Denied'
    else:

        # Node 2: If no derogatory reports, check income per dependent (key indicator of repayment capacity)
        # Condition: income_per_dependent >= 5.0
        if income_per_dependent >= 5.0:
            prediction = 'Accepted'
        else:
            prediction = 'Denied'

    return prediction, node_truth_values


   
# Function 4 (from dt_function4.txt)
   
def dt_function_4(age, income_per_dependent, monthly_credit_card_expenses, owning_a_home, self_employed, number_of_derogatory_reports):
    node_truth_values = [0, 0]
    node_truth_values[0] = int(number_of_derogatory_reports > 0)
    node_truth_values[0] = int(income_per_dependent < 4.0)

    # Node 0 (Root Node - Depth 0): The most critical factor for credit approval is often past financial behavior.
    # Condition: Are there any derogatory reports, indicating past financial mismanagement?
    if number_of_derogatory_reports > 0:
        prediction = "rejected"
    else:  # number_of_derogatory_reports == 0

        # Node 1 (Depth 1, child of the 'no derogatory reports' branch):
        # If there are no derogatory reports, the next most important factor is the applicant's ability to repay.
        # Condition: Is the income per dependent below a reasonable threshold, indicating insufficient repayment capacity?
        # A threshold of 4.0 (on a scale of 1.5 to 10) is chosen as a common "sufficient" level.
        if income_per_dependent < 4.0:
            prediction = "rejected"
        else:  # income_per_dependent >= 4.0
            prediction = "accepted"
            
    return prediction, node_truth_values

