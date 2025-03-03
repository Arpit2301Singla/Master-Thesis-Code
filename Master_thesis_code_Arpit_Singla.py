import numpy as np
import math
import csv
import os
from scipy.integrate import quad
from itertools import product

# [Previous function definitions remain the same until process_mof]
# Function Definitions from thesis_4.py
def calculate_Hads(q, Qst):
    Hads, _ = quad(lambda x: Qst, 0, q)
    return Hads

def calculate_total_energy_required(MOF_W, working_capacity, E_par):
    return MOF_W * (working_capacity / 1000) * E_par

def calculate_delta_Hsen(T_ads, T_des, Cp_m):
    delta_Hsen, _ = quad(lambda T: Cp_m, T_ads, T_des)
    return delta_Hsen

def calculate_working_capacity(q_ads, q_des):
    return (q_ads - q_des) * 44  # Convert mmol/g to mg/g

def calculate_water_effect(H2O_uptake, Cp_H2O, T_ads, T_des):
    if T_des > 373.15:
       T_des = 373.15  # Limit T_des to 100 if it exceeds this value
    delta_T = T_des - T_ads
    H2O_uptake_g = (H2O_uptake / 1000) * 18
    return H2O_uptake_g * Cp_H2O * delta_T

def calculate_parasitic_energy(T_ads, T_des, q_ads, q_des, Qst, Cp_m, MOF_W, H2O_uptake=0, Cp_H2O=4.18):
    Hads_ads = calculate_Hads(q_ads, Qst)
    Hads_des = calculate_Hads(q_des, Qst)
    delta_Hdes = Hads_ads - Hads_des
    delta_Hsen = calculate_delta_Hsen(T_ads, T_des, Cp_m)
    delta_Hwater = calculate_water_effect(H2O_uptake, Cp_H2O, T_ads, T_des)
    E_reg = delta_Hdes + delta_Hsen + delta_Hwater
    working_capacity = calculate_working_capacity(q_ads, q_des)
    E_par = (E_reg / 1000) / (working_capacity / 1000)  # kJ/g per g of CO2
    total_energy_required = calculate_total_energy_required(MOF_W, working_capacity, E_par)
    total_CO2_capture = MOF_W * (working_capacity / 1000)

    return {
        "Hads_adsorption": Hads_ads,
        "Hads_desorption": Hads_des,
        "delta_Hdes": delta_Hdes,
        "delta_Hsen": delta_Hsen,
        "delta_Hwater": delta_Hwater,
        "E_reg": E_reg,
        "working_capacity": working_capacity,
        "E_par": E_par,
        "Total_Energy_Required": total_energy_required,
        "Total_CO2_Capture": total_CO2_capture,
    }

def calculate_airflow_energy(flow_rate_m3h, pressure_drop_pa, adsorption_time_min, fan_efficiency, co2_output_per_cycle_g):
    adsorption_time_h = adsorption_time_min / 60
    co2_output_per_cycle_kg = co2_output_per_cycle_g / 1000
    airflow_energy_kwh_per_kg = ((flow_rate_m3h * pressure_drop_pa * adsorption_time_h) / (fan_efficiency * co2_output_per_cycle_kg)) * 2.7778e-7
    return airflow_energy_kwh_per_kg

def calculate_vacuum_energy(Pext, volume, P1, P2, n):
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be greater than zero.")

    return -(((Pext * (volume/1000))/n) * ((P1/Pext) - (P2/Pext) + math.log(P2/P1))) * 100 * 2.7778e-7 #kwh

def get_input(prompt, default, value_type=float):
    try:
        value = input(f"{prompt} (default={default}): ")
        return value_type(value) if value else default
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return default

def get_range_input(prompt, default, range_steps=10):
    try:
        input_val = input(f"{prompt} (default={default}): ")
        if '-' in input_val:  # Check for range input
            start, end = map(float, input_val.split('-'))
            steps = int(input(f"Enter number of steps for range {start}-{end} (default={range_steps}): ") or range_steps)
            return np.linspace(start, end, steps)
        else:  # Single value
            return np.array([float(input_val)]) if input_val else np.array([default])
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return np.array([default])

def calculate_cost(total_energy, cost_per_kWh):
    return (total_energy) * (cost_per_kWh / 100)

def calculate_emissions(total_energy, emission_factor):
    return (total_energy) * emission_factor

def calculate_mof_cost(MOF_W, MOF_price_per_kg):
    return (MOF_W / 1000) * MOF_price_per_kg

def process_mof(T_ads, T_des, q_ads, q_des, Qst, Cp_m, MOF_W, CO2_capture_needed, H2O_uptake, Cp_H2O, 
                flow_rate_m3h, pressure_drop_pa, adsorption_time_min, fan_efficiency, P1_evac, P2_evac, 
                Pext_evac, P1_desorp, P2_desorp, Pext_desorp, vacuum_efficiency, Vol_unit, cycle_limit, 
                decrement_interval, decrement_percentage, stop_threshold, cost_per_kWh, emission_factor, mof_number):
    
    total_energy = 0
    total_co2_capture = 0
    initial_q_ads = q_ads
    current_q_ads = q_ads
    degradation_history = []  # To track q_ads changes

    for cycle in range(1, int(cycle_limit) + 1):
        if cycle % int(decrement_interval) == 0:
            current_q_ads *= (1 - float(decrement_percentage) / 100)
            degradation_history.append((cycle, current_q_ads))

        if current_q_ads <= initial_q_ads * float(stop_threshold):
            break

        parasitic_results = calculate_parasitic_energy(T_ads, T_des, current_q_ads, q_des, Qst, Cp_m, MOF_W, H2O_uptake, Cp_H2O)
        total_energy += parasitic_results["Total_Energy_Required"]
        total_co2_capture += parasitic_results["Total_CO2_Capture"]

    co2_captured_per_cycle = total_co2_capture / cycle
    Total_heat_energy_per_kg_co2 = (total_energy/3600) / (total_co2_capture/1000)
    Heat_energy_needed = Total_heat_energy_per_kg_co2 * CO2_capture_needed
    airflow_energy_per_kg_co2 = calculate_airflow_energy(flow_rate_m3h, pressure_drop_pa, adsorption_time_min, fan_efficiency, co2_captured_per_cycle)
    Total_airflow_energy = airflow_energy_per_kg_co2 * (co2_captured_per_cycle/1000) * cycle
    Fan_energy_needed = airflow_energy_per_kg_co2 * CO2_capture_needed

    vacuum_evacuation = calculate_vacuum_energy(Pext_evac, Vol_unit, P1_evac, P2_evac, vacuum_efficiency)
    Vacuum_desorption = calculate_vacuum_energy(Pext_desorp, Vol_unit, P1_desorp, P2_desorp, vacuum_efficiency)
    Total_vacuum_energy = (vacuum_evacuation + Vacuum_desorption) * cycle
    Total_vacuum_Energy_per_KG_CO2 = (vacuum_evacuation + Vacuum_desorption) / (co2_captured_per_cycle/1000)
    Vacuum_energy_needed = Total_vacuum_Energy_per_KG_CO2 * CO2_capture_needed

    total_energy_required = (total_energy/3600) + Total_vacuum_energy + Total_airflow_energy
    total_energy_per_kg_co2_capture = Total_heat_energy_per_kg_co2 + airflow_energy_per_kg_co2 + Total_vacuum_Energy_per_KG_CO2
    Total_energy_needed = total_energy_per_kg_co2_capture * CO2_capture_needed

    total_cost_energy = calculate_cost(total_energy_required, cost_per_kWh)
    energy_cost_per_kg_co2 = total_cost_energy/(total_co2_capture/1000)
    energy_cost_needed = energy_cost_per_kg_co2 * CO2_capture_needed
    total_emissions = calculate_emissions(total_energy_required, emission_factor)
    total_cost_mof = calculate_mof_cost(MOF_W, 100) #MOF price
    total_cost_mof_per_kg_co2 = total_cost_mof/(total_co2_capture/1000)
    MOF_cost_needed = total_cost_mof_per_kg_co2 * CO2_capture_needed
    total_cost = total_cost_energy + total_cost_mof
    total_cost_per_kg_co2 = total_cost_mof_per_kg_co2 + energy_cost_per_kg_co2
    total_cost_needed = total_cost_per_kg_co2 * CO2_capture_needed
    MOF_needed = (MOF_W / total_co2_capture) * CO2_capture_needed
    number_of_cycles_needed = (cycle/(MOF_W/1000)) * MOF_needed

    emissions_per_unit_co2 = total_emissions / total_co2_capture if total_co2_capture > 0 else float('inf')

    # Calculate degradation percentage
    final_degradation = ((initial_q_ads - current_q_ads) / initial_q_ads) * 100

    results = {
        # Input Parameters
        'MOF_Number': mof_number,
        'T_ads (K)': T_ads,
        'T_des (K)': T_des,
        'Initial_q_ads (mmol/g)': initial_q_ads,
        'Final_q_ads (mmol/g)': current_q_ads,
        'Total_q_ads_Degradation (%)': final_degradation,
        'q_des (mmol/g)': q_des,
        'Qst (J/mmol)': Qst,
        'Cp_m (J/g.K)': Cp_m,
        'MOF_W in a batch (g)': MOF_W,
        'CO2_capture_needed (kg)': CO2_capture_needed,
        'H2O_uptake (mmol/g)': H2O_uptake,
        'Cp_H2O (J/g.K)': Cp_H2O,
        'flow_rate (m3/h)': flow_rate_m3h,
        'pressure_drop (Pa)': pressure_drop_pa,
        'adsorption_time (min)': adsorption_time_min,
        'fan_efficiency (-)': fan_efficiency,
        'P1_evac (mbar)': P1_evac,
        'P2_evac (mbar)': P2_evac,
        'Pext_evac (mbar)': Pext_evac,
        'P1_desorp (mbar)': P1_desorp,
        'P2_desorp (mbar)': P2_desorp,
        'Pext_desorp (mbar)': Pext_desorp,
        'vacuum_efficiency (-)': vacuum_efficiency,
        'Vol_unit (L)': Vol_unit,
        'cycle_limit (-)': cycle_limit,
        'decrement_interval (cycles)': decrement_interval,
        'decrement_percentage (%)': decrement_percentage,
        'stop_threshold (-)': stop_threshold,
        'cost_per_kWh (cents)': cost_per_kWh,
        'emission_factor (gCO2eq/kWh)': emission_factor,
        
        # Results
        'Number of cycles per batch': cycle,
        'Total_Cycles for desired CO2 capture (-)': number_of_cycles_needed,
        'MOF_Needed for desired CO2 capture (kg)': MOF_needed,
        'Total_Heat_Energy per batch (kWh)': total_energy/3600,
        'Heat_Energy_per_kg_CO2 (kWh/kg-CO2)': Total_heat_energy_per_kg_co2,
        'Heat_Energy for desired CO2 capture (KWh)': Heat_energy_needed,
        'Total_Fan_Energy per batch (kWh)': Total_airflow_energy,
        'Fan_Energy_per_kg_CO2 (kWh/kg-CO2)': airflow_energy_per_kg_co2,
        'Fan_Energy for desired CO2 capture (KWh)': Fan_energy_needed,
        'Vacuum_Energy_Evacuation (kWh/cycle)': vacuum_evacuation,
        'Vacuum_Energy_Desorption (kWh/cycle)': Vacuum_desorption,
        'Total_Vacuum_Energy per batch (kWh)': Total_vacuum_energy,
        'Vacuum_Energy_per_kg_CO2 (kWh/kg-CO2)': Total_vacuum_Energy_per_KG_CO2,
        'Vacuum_Energy for desired CO2 capture (KWh)': Vacuum_energy_needed,
        'Total_Energy_Required per batch (kWh)': total_energy_required,
        'Energy_per_Cycle (kWh/cycle)': total_energy_required/cycle,
        'Total_Energy_per_kg_CO2 (kWh/kg-CO2)': total_energy_per_kg_co2_capture,
        'Total_Energy for desired CO2 capture (KWh)': Total_energy_needed,
        'CO2_Captured_per_Cycle (g/cycle)': co2_captured_per_cycle,
        'Energy_Cost per batch (euro)': total_cost_energy,
        'Energy_cost_per_kg_CO2 (euro/Kg CO2)': energy_cost_per_kg_co2,
        'Energy_Cost for desired CO2 capture (euro)': energy_cost_needed,
        'MOF_Cost per batch (euro)': total_cost_mof,
        'MOF_Cost_per_KG_CO2 (euro/Kg CO2)': total_cost_mof_per_kg_co2, 
        'MOF_Cost for desired CO2 capture (euro)': MOF_cost_needed,
        'Total_Cost per batch (euro)': total_cost,
        'Total_Cost_per_KG_CO2 (euro/Kg CO2)': total_cost_per_kg_co2,
        'Total_cost for desired CO2 capture (euro)': total_cost_needed,
        'Total_CO2_Captured per batch (g)': total_co2_capture,
        'Total_Emissions per batch (gCO2eq)': total_emissions,
        'Emissions_per_CO2_Captured (gCO2eq/g-CO2)': emissions_per_unit_co2
    }

    return results

def main_with_ranges():
    print("=== Integrated Energy Calculation for Multiple MOFs with Ranges ===")
    
    # Energy source selection
    print("\nSelect an energy source:")
    print("1. Renewables")
    print("2. Coal")
    print("3. Nuclear")

    selected_source = "renewables"
    energy_sources = {
        "renewables": {"cost": 9, "emission_factor": 30},
        "coal": {"cost": 22, "emission_factor": 820},
        "nuclear": {"cost": 30, "emission_factor": 12},
    }

    try:
        choice = int(input("Enter your choice (1/2/3): "))
        if choice == 1:
            selected_source = "renewables"
        elif choice == 2:
            selected_source = "coal"
        elif choice == 3:
            selected_source = "nuclear"
    except ValueError:
        print("Invalid input. Defaulting to 'renewables'.")

    cost_per_kWh = energy_sources[selected_source]["cost"]
    emission_factor = energy_sources[selected_source]["emission_factor"]

    num_mofs = get_input("Enter the number of MOFs to process", 1, int)
    all_results = []

    for i in range(1, num_mofs + 1):
        print(f"\n=== Processing MOF {i} ===")
        # [Get all input parameters as before]
        T_ads_range = get_range_input("Enter adsorption temperature (K or range 'start-end')", 298.15)
        T_des_range = get_range_input("Enter desorption temperature (K or range 'start-end')", 373.15)
        q_ads_range = get_range_input("Enter CO2 uptake at adsorption (mmol/g or range)", 0.14)
        q_des_range = get_range_input("Enter CO2 uptake at desorption (mmol/g or range)", 0.0)
        Qst_range = get_range_input("Enter isosteric heat of adsorption (J/mmol or range)", 40.0)
        Cp_m_range = get_range_input("Enter specific heat capacity of the MOF (J/g.K or range)", 0.8)
        MOF_W_range = get_range_input("Enter weight of MOF in a batch (g or range)", 500)
        CO2_capture_needed_range = get_range_input("Enter the amount of CO2 needed to be captured (Kg or range)", 1000)
        H2O_uptake_range = get_range_input("Enter water uptake (mmol/g or range)", 0.1)
        Cp_H2O_range = get_range_input("Enter specific heat capacity of water (J/g.K or range)", 4.18)
        flow_rate_m3h_range = get_range_input("Enter flow rate (m3/h or range)", 50)
        pressure_drop_pa_range = get_range_input("Enter pressure drop (Pa or range)", 100)
        adsorption_time_min_range = get_range_input("Enter adsorption time (minutes or range)", 60)
        fan_efficiency_range = get_range_input("Enter fan efficiency (0-1 or range)", 0.8)
        P1_evac_range = get_range_input("Enter starting pressure for evacuation (mbar or range)", 1000)
        P2_evac_range = get_range_input("Enter ending pressure for evacuation (mbar or range)", 37)
        Pext_evac_range = get_range_input("Enter external pressure for evacuation (mbar or range)", 1000)
        P1_desorp_range = get_range_input("Enter starting pressure for desorption (mbar or range)", 264)
        P2_desorp_range = get_range_input("Enter ending pressure for desorption (mbar or range)", 60)
        Pext_desorp_range = get_range_input("Enter external pressure for desorption (mbar or range)", 1000)
        vacuum_efficiency_range = get_range_input("Enter vacuum efficiency (0-1 or range)", 0.7)
        Vol_unit_range = get_range_input("Enter volume of the unit (Liters or range)", 7)
        cycle_limit_range = get_range_input("Enter cycle limit (or range 'start-end')", 10000)
        decrement_interval_range = get_range_input("Enter decrement interval (or range 'start-end')", 100)
        decrement_percentage_range = get_range_input("Enter decrement percentage (or range 'start-end')", 5.0)
        stop_threshold_range = get_range_input("Enter stop threshold (or range 'start-end')", 0.5)

        # Generate all combinations using itertools.product
        parameter_combinations = product(
            T_ads_range, T_des_range, q_ads_range, q_des_range, Qst_range, Cp_m_range,
            MOF_W_range, CO2_capture_needed_range, H2O_uptake_range, Cp_H2O_range,
            flow_rate_m3h_range, pressure_drop_pa_range, adsorption_time_min_range,
            fan_efficiency_range, P1_evac_range, P2_evac_range, Pext_evac_range,
            P1_desorp_range, P2_desorp_range, Pext_desorp_range, vacuum_efficiency_range,
            Vol_unit_range, cycle_limit_range, decrement_interval_range, decrement_percentage_range,
            stop_threshold_range
        )

        for params in parameter_combinations:
            results = process_mof(
                *params,
                cost_per_kWh=cost_per_kWh,
                emission_factor=emission_factor,
                mof_number=i
            )
            all_results.append(results)

    # Write results to CSV
    if all_results:
        filename = f'mof_analysis_results_{selected_source}.csv'
        
        # First, write the file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            
            # Then, modify the file permissions to ensure it's not read-only
            if os.name == 'nt':  # For Windows
                os.chmod(filename, 0o666)  # Read and write for owner, group, and others
            else:  # For Unix-like systems
                current_permissions = os.stat(filename).st_mode
                os.chmod(filename, current_permissions | 0o666)  # Add read/write permissions
                
            print(f"\nResults have been saved to {filename}")
            
        except PermissionError as e:
            print(f"Permission error while creating file: {e}")
            print("Try running the script with appropriate permissions or in a different directory.")
        except Exception as e:
            print(f"An error occurred while creating the file: {e}")

if __name__ == "__main__":
    main_with_ranges()
