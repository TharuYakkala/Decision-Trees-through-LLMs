"""
bankruptcy - 5 Zero-Shot Decision Tree Functions (Gemini 2.5 Flash)
"""

# Function 0 (from dt_function0.txt)
def dt_function_0(WC_TA, RE_TA, EBIT_TA, S_TA, BVE_BVL):
    """
    Predicts if a telecommunications firm will declare bankruptcy based on financial ratios.

    This decision tree is induced based on expert domain knowledge without real-world data,
    focusing on the most impactful features for bankruptcy prediction (solvency, profitability, liquidity).
    The tree has a maximum depth of 2.

    Args:
        WC_TA (float): Working Capital as a percentage of Total Assets.
        RE_TA (float): Retained Earnings as a percentage of Total Assets.
        EBIT_TA (float): Earnings Before Interest and Tax as a percentage of Total Assets.
        S_TA (float): Sales as a percentage of Total Assets.
        BVE_BVL (float): Book Value of Equity divided by Book Value of Total Liabilities.

    Returns:
        tuple: A tuple containing:
            - str: The predicted class ("Bankrupt" or "Non-Bankrupt").
            - list: A list of 1s and 0s representing the truth values of the inner nodes.
                    The order of nodes is:
                    [0] Root node: (BVE_BVL <= 0.75)
                    [1] Left child of root: (EBIT_TA <= 0.0)
                    [2] Right child of root: (WC_TA <= 5.0)
                    A 1 indicates the condition was satisfied, 0 otherwise.
    """
    # Initialize list for truth values of the three inner nodes:
    # 0: Condition for the root node (BVE_BVL <= 0.75)
    # 1: Condition for the left child node (EBIT_TA <= 0.0)
    # 2: Condition for the right child node (WC_TA <= 5.0)
    node_truth_values = [0, 0, 0]
    node_truth_values[0] = int(BVE_BVL <= 0.75)
    node_truth_values[1] = int(EBIT_TA <= 0.0)
    node_truth_values[2] = int(WC_TA <= 5.0)
    prediction = ""

    # Node 0 (Root): Check Book Value of Equity / Book Value of Liabilities (BVE/BVL)
    # A low BVE/BVL (e.g., <= 0.75, meaning liabilities are at least 1.33 times equity)
    # indicates high leverage and significant solvency risk.
    if node_truth_values[0]:
        # Node 1 (Left Child): Check Earnings Before Interest and Tax / Total Assets (EBIT/TA)
        # If highly leveraged and not generating operational profit (EBIT <= 0), the firm is in severe distress.
        if node_truth_values[1]:
            prediction = "Bankrupt"
        else:
            # Even with high leverage, if operationally profitable, there might be a chance for survival.
            # This represents a less immediate risk compared to non-profitability under high leverage.
            prediction = "Non-Bankrupt"
    else:

        # Node 2 (Right Child): Check Working Capital / Total Assets (WC/TA)
        # Even with healthy leverage, a firm can face bankruptcy due to severe liquidity problems.
        # A low WC/TA (e.g., <= 5.0%) suggests insufficient working capital.
        if node_truth_values[2]:  # Assuming WC/TA is expressed as a percentage
            prediction = "Bankrupt"
        else:
            # With both lower leverage and healthy liquidity, the firm is likely stable.
            prediction = "Non-Bankrupt"

    return prediction, node_truth_values


 
# Function 1 (from dt_function1.txt)
  
def dt_function_1(features):
    """
    Predicts if a telecommunications firm will go bankrupt based on financial ratios.

    Args:
        features (dict): A dictionary containing the following financial ratios:
            'Working Capital/Total Assets' (float, percentage)
            'Retained Earnings/Total Assets' (float, percentage)
            'Earnings Before Interest and Tax/Total Assets' (float, percentage)
            'Sales/Total Assets' (float, percentage)
            'Book Value of Equity/Book Value of Liabilities' (float)

    Returns:
        tuple: A tuple containing:
            - int: The prediction (1 for Bankrupt, 0 for Non-Bankrupt).
            - list: A list of truth values (1 if condition satisfied, 0 otherwise)
                    for the inner nodes traversed.
    """
    # Extract features using the exact names provided
    wc_ta = features['Working Capital/Total Assets']
    # re_ta = features['Retained Earnings/Total Assets'] # Not used in this specific tree
    ebit_ta = features['Earnings Before Interest and Tax/Total Assets']
    # s_ta = features['Sales/Total Assets'] # Not used in this specific tree
    bve_bvl = features['Book Value of Equity/Book Value of Liabilities']

    # Initialize list to store truth values for inner nodes
    prediction = -1 # Default, should be overwritten

    # Root Node (Depth 0): Book Value of Equity / Book Value of Liabilities (BVE/BVL)
    # This ratio measures financial leverage. A value <= 0.5 indicates that equity is
    # half or less than total liabilities, which is a strong indicator of financial distress.
    condition_bve_bvl_low = int(bve_bvl <= 0.5)
    condition_ebit_ta_negative = int(ebit_ta <= 0.0)
    condition_wc_ta_negative = int(wc_ta <= 0.0)
    node_truth_values = [condition_bve_bvl_low, condition_ebit_ta_negative, condition_wc_ta_negative]

    if condition_bve_bvl_low:
        # Path 1: High Leverage / Low Equity (BVE/BVL <= 0.5) - This path already suggests high risk.
        # Node 1.1 (Depth 1): Earnings Before Interest and Tax / Total Assets (EBIT/TA)
        # This measures operational profitability. A value <= 0.0 means the company is
        # operating at a loss or just breaking even, which is critical when combined with high leverage.
        condition_ebit_ta_negative = (ebit_ta <= 0.0)

        if condition_ebit_ta_negative:
            # Prediction: Bankrupt (Extremely high risk: High leverage AND negative operational profitability)
            prediction = 1
        else:
            # Prediction: Non-Bankrupt (Still high leverage, but at least operationally profitable.
            # There's a chance for recovery or to sustain operations for a while).
            prediction = 0
    else:
        # Path 2: Acceptable Leverage (BVE/BVL > 0.5) - Less immediate risk from solvency perspective.
        # Node 1.2 (Depth 1): Working Capital / Total Assets (WC/TA)
        # This measures short-term liquidity. A value <= 0.0 indicates severe liquidity issues,
        # meaning current liabilities exceed current assets. Even with good leverage, poor liquidity can lead to bankruptcy.

        if condition_wc_ta_negative:
            # Prediction: Bankrupt (Leverage is okay, BUT facing a severe short-term liquidity crisis)
            prediction = 1
        else:
            # Prediction: Non-Bankrupt (Leverage is acceptable AND liquidity is healthy)
            prediction = 0
            
    return prediction, node_truth_values



# Function 2 (from dt_function2.txt)
def dt_function_2(WC_TA, RE_TA, EBIT_TA, S_TA, BVE_BVL):
    """
    Classifies a telecommunications firm as 'Bankrupt' or 'Not Bankrupt'
    based on financial ratios, using a depth-2 decision tree.

    Features (all expressed as percentages or ratios):
    WC_TA (Working Capital/Total Assets)
    RE_TA (Retained Earnings/Total Assets)
    EBIT_TA (Earnings Before Interest and Tax/Total Assets)
    S_TA (Sales/Total Assets)
    BVE_BVL (Book Value of Equity/Book Value of Liabilities)

    Returns:
    - prediction (str): 'Bankrupt' or 'Not Bankrupt'
    - truth_values (list): A list of 1s and 0s representing the truth values
                          of the inner node conditions.
                          Index 0: BVE_BVL <= 0.1
                          Index 1: EBIT_TA <= -10
                          Index 2: WC_TA <= 0
    """
    
    # Initialize truth values for all inner nodes to 0.
    # The order corresponds to: [Node 0: BVE_BVL, Node 1: EBIT_TA, Node 2: WC_TA]
    truth_values = [0, 0, 0] 
    truth_values[0] = int(BVE_BVL <= 0.1)
    truth_values[1] = int(EBIT_TA <= -10)
    truth_values[2] = int(WC_TA <= 0)
    prediction = "Not Bankrupt" # Default prediction for the least risky path

    # Node 0: Root node - Check solvency/leverage (Book Value of Equity / Book Value of Liabilities)
    # A very low BVE/BVL (e.g., 0.1 or less) indicates severe equity erosion and high financial distress.
    if BVE_BVL <= 0.1:
        # Branch Left: Firm is highly distressed due to poor solvency
        # Node 1: Check operational profitability (Earnings Before Interest and Taxes / Total Assets)
        # Even within a highly leveraged firm, severe operational losses accelerate bankruptcy.
        if EBIT_TA <= -10: # Significant operational losses (e.g., losing more than 10% of assets in earnings)
            prediction = "Bankrupt"
        else: # Operational profits are not catastrophically negative, but still highly distressed from solvency.
            prediction = "Bankrupt" # Still predicts bankrupt due to the severe BVE_BVL condition
    else:
        truth_values[0] = 0 # Condition for Node 0 is not satisfied
        
        # Branch Right: Firm has better solvency, but can still face issues
        # Node 2: Check liquidity (Working Capital / Total Assets)
        # Even with decent equity, a company can go bankrupt if it cannot meet short-term obligations.
        if WC_TA <= 0: # Negative working capital indicates current liabilities exceed current assets (liquidity crisis)
            prediction = "Bankrupt"
        else: # Positive working capital indicates better liquidity.
            prediction = "Not Bankrupt" # Firm appears relatively healthy based on these key indicators

    return prediction, truth_values


# Function 3 (from dt_function3.txt)
def dt_function_3(
    Working_Capital_Total_Assets: float,
    Retained_Earnings_Total_Assets: float,
    Earnings_Before_Interest_and_Tax_Total_Assets: float,
    Sales_Total_Assets: float,
    Book_Value_of_Equity_Book_Value_of_Liabilities: float
) -> tuple[str, list[int]]:
    """
    Classifies a telecommunications firm as 'Bankrupt' or 'Non-Bankrupt' based on financial ratios.

    Args:
        Working_Capital_Total_Assets (float): Working capital as a percentage of total assets.
        Retained_Earnings_Total_Assets (float): Retained earnings as a percentage of total assets.
        Earnings_Before_Interest_and_Tax_Total_Assets (float): Earnings before interest and taxes as a percentage of total assets.
        Sales_Total_Assets (float): Sales as a percentage of total assets.
        Book_Value_of_Equity_Book_Value_of_Liabilities (float): Book value of equity divided by book value of total liabilities.

    Returns:
        tuple[str, list[int]]: A tuple containing the prediction ('Bankrupt' or 'Non-Bankrupt')
                               and a list representing the truth values of the inner nodes traversed.
                               Each entry in the list is 1 if the condition of the corresponding inner node
                               is satisfied, and 0 otherwise.
    """
    truth_values = [0, 0]
    condition_node_0 = (Book_Value_of_Equity_Book_Value_of_Liabilities <= 0.2)
    condition_node_1 = (Earnings_Before_Interest_and_Tax_Total_Assets <= 0)
    truth_values[0] = int(condition_node_0)
    truth_values[1] = int(condition_node_1)

    # Node 0: Book Value of Equity / Book Value of Liabilities (BVE/BVL) <= 0.2
    # Rationale: A very low BVE/BVL ratio (equity less than 20% of liabilities, or even negative)
    # is a critical indicator of severe financial distress and high insolvency risk.

    if condition_node_0:
        # If BVE/BVL is critically low, the firm is highly likely to be Bankrupt.
        # This acts as an early exit for severe cases.
        prediction = "Bankrupt"
    else:
        # If BVE/BVL is above the critical threshold, assess current operational profitability.
        # Node 1 (right child of Node 0): Earnings Before Interest and Tax / Total Assets (EBIT/TA) <= 0
        # Rationale: Even if solvency appears somewhat stable, consistently operating at a loss
        # (negative or zero EBIT/TA) indicates an unsustainable business model and high risk of future bankruptcy.

        if condition_node_1:
            prediction = "Bankrupt"
        else:
            # If BVE/BVL is not critically low and the firm is operationally profitable,
            # it indicates a healthier financial state.
            prediction = "Non-Bankrupt"

    return prediction, truth_values


# Function 4 (from dt_function4.txt)
def dt_function_4(We_TA, RE_TA, EBIT_TA, S_TA, BVE_BVL):
    """
    Classifies a telecommunications firm as 'BANKRUPT' or 'NON-BANKRUPT'
    based on key financial ratios using an expert-induced decision tree of maximum depth 2.

    Parameters:
    We_TA (float): Working Capital as a percentage of Total Assets (e.g., 0.10 for 10%).
    RE_TA (float): Retained Earnings as a percentage of Total Assets (e.g., -0.05 for -5%).
    EBIT_TA (float): Earnings Before Interest and Tax as a percentage of Total Assets (e.g., 0.02 for 2%).
    S_TA (float): Sales as a percentage of Total Assets (e.g., 1.50 for 150%).
    BVE_BVL (float): Book Value of Equity divided by Book Value of Liabilities (e.g., 0.4 for 0.4).

    Returns:
    tuple: (prediction, truth_values)
        prediction (str): 'BANKRUPT' or 'NON-BANKRUPT'.
        truth_values (list): A list representing the truth values of the inner nodes encountered
                             during classification. Each entry is 1 if the condition of the
                             corresponding inner node is satisfied, and 0 otherwise.
                             The nodes are evaluated in the following logical order:
                             1. BVE_BVL <= 0.5 (Root node)
                             2. EBIT_TA <= 0.0 (if Root node's TRUE branch is taken)
                             3. RE_TA <= -0.1 (if Root node's FALSE branch is taken)
    """
    truth_values = [0, 0, 0]
    truth_values[0] = int(BVE_BVL <= 0.5)
    truth_values[1] = int(EBIT_TA <= 0.0)
    truth_values[2] = int(RE_TA <= -0.1)
    prediction = ""

    # Node 0 (Root Node): Book Value of Equity / Book Value of Liabilities (BVE/BVL)
    # This is a primary indicator of solvency. A ratio of 0.5 or less implies that
    # the company's equity is half or less of its liabilities, signaling high leverage
    # and a significant risk of insolvency.
    if BVE_BVL <= 0.5:
        # Node 1 (Child of Root TRUE branch): Earnings Before Interest and Tax / Total Assets (EBIT/TA)
        # For a highly leveraged firm, current operational profitability is critical.
        # If it's operating at a loss (EBIT/TA <= 0), bankruptcy risk is extremely high.
        if EBIT_TA <= 0.0:
            prediction = "BANKRUPT"
        else:
            # A highly leveraged firm that is still generating positive operating earnings
            # might have a chance to service its debt and avoid bankruptcy, though still risky.
            prediction = "NON-BANKRUPT"
    else:
        # Node 2 (Child of Root FALSE branch): Retained Earnings / Total Assets (RE/TA)
        # For firms with healthier leverage, cumulative profitability is a key indicator.
        # Significant accumulated losses (RE/TA <= -0.1, meaning 10% or more of assets lost)
        # can deplete capital and lead to bankruptcy even if current leverage is moderate.
        if RE_TA <= -0.1:
            prediction = "BANKRUPT"
        else:
            # A firm with healthier leverage and no significant accumulated losses is
            # considered financially stable in this model.
            prediction = "NON-BANKRUPT"

    return prediction, truth_values

