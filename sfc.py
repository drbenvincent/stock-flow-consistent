import sympy as sp
import numpy as np
import pandas as pd


def solve(
    godley_table, params_symbols, initial_values, params, t0=0, t_end=500, dt=0.1
) -> pd.DataFrame:
    """Solve the stock-flow consistent model using Euler's method."""
    
    times = np.arange(t0, t_end, dt)  # Create an array of time steps
    num_steps = len(times)

    t = sp.symbols("t")  # time

    # Generate the differential equations - basically column sums of the godley table
    diff_eqs = generate_differential_equations(godley_table, t)
    # Generate the derivative functions
    derivatives = create_derivative_functions(diff_eqs, params_symbols)

    # Initialize arrays to store stock values over time
    stock_values = {sector: np.zeros(num_steps) for sector in initial_values.keys()}

    # Set the initial stock values
    for sector in initial_values:
        stock_values[sector][0] = initial_values[sector]

    # Perform the simulation using Euler's method
    for i in range(1, num_steps):
        # Compute the derivative for each sector
        for sector, deriv_func in derivatives.items():
            # Extract parameter values from the params dictionary
            param_values = [params[key] for key in params.keys()]
            # Calculate the derivative using the parameter values from the dictionary
            derivative_value = deriv_func(*param_values)
            # Update the stock using Euler's method
            stock_values[sector][i] = (
                stock_values[sector][i - 1] + dt * derivative_value
            )

    return pd.DataFrame(stock_values, index=times)


def generate_differential_equations(godley_table, t: sp.Symbol):
    # Create a dictionary to store the stock variables for each sector
    stocks = {}

    # Create symbolic stock variables for each sector (e.g., Household Savings, Government Debt)
    for sector in godley_table.columns:
        stocks[sector] = sp.Function(f"S_{sector}")(t)

    # List to hold differential equations
    diff_eqs = {}

    # For each sector, sum the flows (rows) and generate the differential equation for the stock
    for sector in godley_table.columns:
        net_flow = sum(godley_table[sector])  # Sum the flows for the sector
        diff_eq = sp.Eq(
            stocks[sector].diff(t), net_flow
        )  # Create the differential equation
        diff_eqs[sector] = diff_eq

    return diff_eqs


def create_derivative_functions(diff_eqs, params_symbols: list):
    derivatives = {}
    for sector, eq in diff_eqs.items():
        # Get the right-hand side of the equation (the net flow)
        rhs = eq.rhs
        # Convert the symbolic expression into a callable Python function
        derivatives[sector] = sp.lambdify(params_symbols, rhs)
    return derivatives


def extract_symbols_from_flows(flows) -> list[sp.Symbol]:
    """Extract all sympy symbols from the flows, ignoring signs and handling zeros"""
    all_symbols = set()

    # Extract all equations from the flows and flatten the list
    equations = [expr for flow in flows for expr in flow[1]]

    for expr in equations:
        if expr != 0:  # Only extract symbols if the expression is non-zero
            all_symbols.update(expr.free_symbols)

    return list(all_symbols)
