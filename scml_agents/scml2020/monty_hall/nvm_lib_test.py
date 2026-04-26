# Needs to be called just once in the constructor of the agent
from .nvm_lib.nvm_lib import NVMLib

if __name__ == "__main__":
    nvm = NVMLib(
        mpnvp_number_of_periods=3,
        mpnvp_quantities_domain_size=10,
        game_length=50,
        input_product_index=1,
        output_product_index=2,
        num_intermediate_products=3,
        production_cost=1.0,
    )

    # Need to be called at each simulation time. Returns the plan for step current_time. Verbose to get some info into what is going on.
    nvm_sol = nvm.get_complete_plan(current_time=30, verbose=True)
    print(f"nvm_sol = {nvm_sol}")
