import unittest
from unittest.mock import MagicMock, patch
import os
import uuid

# Assuming your agent and inventory manager are in these files
# Adjust the import path if necessary
from inventory_manager_cir import (
    InventoryManagerCIR,
    MaterialType,
    IMContractType,
    IMContract,
    Batch,
)
from .litaagent_cir import LitaAgentCIR  # Assuming this is your agent class

# Constants from SCML or your agent (ensure these match your agent's usage)
QUANTITY = 0
TIME = 1
UNIT_PRICE = 2


# Helper function to create a mock AWI
def create_mock_awi(
    current_step=0,
    n_steps=20,
    is_first_level=False,
    is_last_level=False,
    my_suppliers=None,
    my_consumers=None,
):
    awi = MagicMock()
    awi.current_step = current_step
    awi.n_steps = n_steps
    awi.is_first_level = is_first_level
    awi.is_last_level = is_last_level
    awi.my_suppliers = my_suppliers if my_suppliers is not None else []
    awi.my_consumers = my_consumers if my_consumers is not None else []

    # Mock other AWI attributes/methods as needed by your agent
    awi.current_exogenous_input_quantity = 0
    awi.current_exogenous_output_quantity = 0
    # ...

    # Mock NMI related parts if AWI is responsible for providing them directly
    # For LitaAgentCIR, it seems get_nmi is a method of the agent itself,
    # and it might get NMI data from self.negotiator_nmi_cache or similar.
    # So, we might need to mock that cache or the underlying mechanism.
    return awi


# Helper function to create a mock NMI
def create_mock_nmi(q_min=1, q_max=100, p_min=1.0, p_max=20.0, t_min=1, t_max=10):
    nmi = MagicMock()
    nmi.issues = {
        QUANTITY: MagicMock(min_value=q_min, max_value=q_max)
        if q_min is not None
        else None,
        UNIT_PRICE: MagicMock(min_value=p_min, max_value=p_max)
        if p_min is not None
        else None,
        TIME: MagicMock(min_value=t_min, max_value=t_max)
        if t_min is not None
        else None,
    }
    return nmi


class TestInventoryManagerCIR(unittest.TestCase):
    def setUp(self):
        # Create env.test to enable debug prints if your code uses it
        with open("env.test", "w") as f:
            pass
        self.im = InventoryManagerCIR(
            raw_storage_cost=0.1,
            product_storage_cost=0.2,
            processing_cost=1.0,
            daily_production_capacity=10,
            max_simulation_day=20,
            current_day=0,
        )

    def tearDown(self):
        # Clean up env.test
        if os.path.exists("env.test"):
            os.remove("env.test")

    # --- 1.1 Initialization ---
    def test_IM_Init_Normal(self):
        self.assertEqual(self.im.current_day, 0)
        self.assertEqual(self.im.raw_storage_cost_per_unit_per_day, 0.1)
        self.assertEqual(self.im.daily_production_capacity, 10)

    # --- 1.2 Inventory Updates ---
    def test_IM_AddMaterial_And_Consume(self):
        batch_id = str(uuid.uuid4())
        self.im.raw_material_batches.append(
            Batch(batch_id, 100, 100, 5.0, 0, MaterialType.RAW)
        )
        self.assertEqual(
            sum(b.remaining_quantity for b in self.im.raw_material_batches), 100
        )

        # Simulate production consuming materials
        # This part would be more realistically tested via _execute_production
        # For a direct test of batch update:
        self.im.raw_material_batches[0].remaining_quantity -= 20
        self.assertEqual(self.im.raw_material_batches[0].remaining_quantity, 80)

    def test_IM_ConsumeMaterial_Insufficient(self):
        batch_id = str(uuid.uuid4())
        self.im.raw_material_batches.append(
            Batch(batch_id, 10, 10, 5.0, 0, MaterialType.RAW)
        )

        # Attempt to consume more than available (directly manipulating for test simplicity)
        # In real use, _execute_production handles this.
        consumed_qty = 0
        qty_to_consume = 15
        temp_batches_to_remove = []
        for i, batch in enumerate(self.im.raw_material_batches):
            can_consume = min(qty_to_consume - consumed_qty, batch.remaining_quantity)
            batch.remaining_quantity -= can_consume
            consumed_qty += can_consume
            if batch.remaining_quantity <= 0:
                temp_batches_to_remove.append(i)

        for i in sorted(temp_batches_to_remove, reverse=True):
            del self.im.raw_material_batches[i]

        self.assertEqual(consumed_qty, 10)
        self.assertEqual(len(self.im.raw_material_batches), 0)

    # --- 1.3 Production Planning & Execution ---
    def test_IM_PlanProduction_Normal(self):
        # Add a demand contract
        self.im.add_transaction(
            IMContract(
                "C1",
                str(uuid.uuid4()),
                IMContractType.DEMAND,
                15,
                10.0,
                5,
                MaterialType.PRODUCT,
            )
        )
        # plan_production is called within add_transaction
        # Expected: produce 10 on day 5, 5 on day 4 (due to capacity 10)
        self.assertEqual(self.im.production_plan.get(5, 0), 10)
        self.assertEqual(self.im.production_plan.get(4, 0), 5)

    def test_IM_ExecuteProduction_Normal(self):
        self.im.raw_material_batches.append(
            Batch(str(uuid.uuid4()), 20, 20, 1.0, 0, MaterialType.RAW)
        )
        self.im.production_plan = {0: 5}  # Plan to produce 5 on day 0
        self.im._execute_production(0)
        self.assertEqual(
            sum(
                b.remaining_quantity
                for b in self.im.product_batches
                if b.arrival_or_production_time == 0
            ),
            5,
        )
        self.assertEqual(
            sum(b.remaining_quantity for b in self.im.raw_material_batches), 15
        )

    # --- 1.4 Cost Calculation (Simplified example) ---
    def test_IM_StorageCost_Raw_NonEmpty(self):
        self.im.raw_material_batches.append(
            Batch(str(uuid.uuid4()), 10, 10, 1.0, 0, MaterialType.RAW)
        )
        # This test would be more complex, needing to simulate days passing
        # For now, we'll focus on the structure.
        # A full cost calculation test would involve calling a method that sums up costs over days.
        # Let's assume a get_daily_storage_cost method for this example
        # For day 1, storage cost for 10 units from day 0 = 10 * 0.1 * 1 = 1.0
        # This requires a method like calculate_daily_storage_costs(day)
        pass  # Placeholder for more detailed cost tests

    # --- 1.5 Contract Handling ---
    def test_IM_AddSupplyContract_FutureDelivery(self):
        contract = IMContract(
            "S1", str(uuid.uuid4()), IMContractType.SUPPLY, 50, 2.0, 3, MaterialType.RAW
        )
        self.im.add_transaction(contract)
        self.assertIn(contract, self.im.pending_supply_contracts)

    def test_IM_ProcessContracts_SupplyArrival(self):
        day_to_process = 2
        self.im.current_day = day_to_process  # Align current day for _receive_materials
        contract = IMContract(
            "S1",
            str(uuid.uuid4()),
            IMContractType.SUPPLY,
            50,
            2.0,
            day_to_process,
            MaterialType.RAW,
        )
        self.im.pending_supply_contracts.append(contract)
        self.im._receive_materials(day_to_process)
        self.assertEqual(len(self.im.pending_supply_contracts), 0)
        self.assertEqual(
            sum(b.remaining_quantity for b in self.im.raw_material_batches), 50
        )

    def test_IM_GetTotalInsufficient_WithShortage(self):
        self.im.production_plan = {2: 20, 3: 15}  # Need 20 on day 2, 15 on day 3
        # No raw materials initially
        shortfall = self.im.get_total_insufficient_raw(
            target_day=2, horizon=2
        )  # Check for day 2 and 3
        self.assertEqual(shortfall, 35)  # 20 (day 2) + 15 (day 3)

        # Add some supply arriving on day 2
        self.im.pending_supply_contracts.append(
            IMContract(
                "S1",
                str(uuid.uuid4()),
                IMContractType.SUPPLY,
                10,
                1.0,
                2,
                MaterialType.RAW,
            )
        )
        shortfall_after_supply = self.im.get_total_insufficient_raw(
            target_day=2, horizon=2
        )
        # Day 2: Need 20, Have 10 (arriving on day 2). Shortfall = 10. Stock becomes 0.
        # Day 3: Need 15, Have 0. Shortfall = 15.
        # Total = 10 + 15 = 25
        self.assertEqual(shortfall_after_supply, 25)


class TestLitaAgentCIR(unittest.TestCase):
    def setUp(self):
        with open("env.test", "w") as f:
            pass

        self.mock_awi = create_mock_awi(current_step=0, n_steps=20)

        # Initialize agent with a mock AWI
        # We need to provide all required __init__ params for LitaAgentCIR
        self.agent = LitaAgentCIR(
            id="test_agent",
            name="TestAgent",
            awi=self.mock_awi,
            # Add other necessary params for LitaAgentCIR constructor
            # For example, if it takes p_threshold, q_threshold directly:
            p_threshold=0.7,
            q_threshold=0.0,
            # If it initializes its own IM, we might need to mock that for some tests
            # or let it initialize and then manipulate its IM.
            # For simplicity, let's assume it initializes its own IM based on AWI.
            # We'll need to ensure the IM gets reasonable default costs.
            raw_storage_cost=0.01,
            product_storage_cost=0.02,
            processing_cost=2.0,
            # daily_production_capacity will be taken from awi if not passed
        )
        # Ensure the agent's IM is initialized for tests that need it
        # If LitaAgentCIR's __init__ doesn't create self.im, you'll need to do it here
        # or ensure on_inventory_update/before_step is called.
        # For now, let's assume a basic IM is created or accessible.
        if not hasattr(self.agent, "im") or self.agent.im is None:
            self.agent.im = InventoryManagerCIR(0.01, 0.02, 2.0, 10, 20, 0)

        # Mock the NMI cache or getter for the agent
        self.agent.negotiator_nmi_cache = {}  # If agent uses such a cache
        # Or mock the get_nmi method directly if it's complex
        self.agent.get_nmi = MagicMock(side_effect=self._get_mock_nmi_for_partner)

        # Mock total_insufficient for relevant tests
        self.agent.total_insufficient = 0
        # Mock market prices if used by first_proposals
        self.agent._market_material_price_avg = 10.0
        self.agent._market_product_price_avg = 20.0

    def tearDown(self):
        if os.path.exists("env.test"):
            os.remove("env.test")

    def _get_mock_nmi_for_partner(self, partner_id):
        # Helper to return specific NMIs for tests
        if partner_id == "supplier1":
            return create_mock_nmi(
                q_min=10, q_max=50, p_min=5.0, p_max=10.0, t_min=1, t_max=5
            )
        elif partner_id == "consumer1":
            return create_mock_nmi(
                q_min=5, q_max=30, p_min=15.0, p_max=25.0, t_min=2, t_max=6
            )
        return create_mock_nmi()  # Default generic NMI

    # --- 2.2 NMI Handling ---
    def test_Agent_GetNMI_ValidPartner(self):
        nmi = self.agent.get_nmi("supplier1")
        self.assertIsNotNone(nmi)
        self.assertEqual(nmi.issues[QUANTITY].min_value, 10)

    # --- 2.3 Scoring functions (Conceptual - requires mocking IM state and offers) ---
    # These tests are complex as score_offers depends heavily on IM state.
    # We'd typically set up specific IM states and then score offers.

    @patch.object(
        InventoryManagerCIR, "calculate_inventory_cost_score"
    )  # Mock the deep calculation
    def test_Score_SingleOffer_Profitable_GoodInventory(self, mock_calc_cost_score):
        # Mock return values: score_a (base_cost), score_b (cost_with_offers)
        # For a profitable, good inventory offer, cost_with_offers < base_cost
        mock_calc_cost_score.side_effect = [1000, 800]  # base_cost, cost_with_offers

        test_offers = {"supplier1": (20, 2, 6.0)}  # Q, T, P

        # We need to ensure the agent's IM is in a state where this offer makes sense.
        # For this isolated test of score_offers, we primarily test the logic flow.
        # The profit normalization part also needs NMI.
        self.agent.negotiator_nmi_cache["supplier1"] = self._get_mock_nmi_for_partner(
            "supplier1"
        )

        raw_score, norm_score, norm_profit = self.agent._evaluate_offer_combinations(
            test_offers, self.agent.im, self.agent.awi
        )
        # This test is for _evaluate_offer_combinations which returns the best combo.
        # Let's test score_offers directly if possible, or ensure _evaluate_offer_combinations
        # correctly calls it and processes results.

        # If testing score_offers directly:
        # raw_score, norm_score = self.agent.score_offers(test_offers, self.agent.im, self.agent.awi)
        # self.assertGreater(raw_score, 0) # Cost reduction
        # self.assertGreater(norm_score, 0.5) # Example, depends on normalization

        # For _evaluate_offer_combinations, it returns (best_combo_items, norm_score, norm_profit)
        # Here, test_offers is the only combo.
        self.assertIsNotNone(raw_score)  # raw_score is best_combo_items here
        self.assertGreater(norm_score, 0.5)  # Assuming good inventory score
        self.assertGreater(
            norm_profit, 0
        )  # Profitable based on NMI (6.0 is good for p_min=5, p_max=10)

    # --- 2.4 first_proposals Method ---
    def test_FP_Procurement_SingleSupplier(self):
        self.agent.total_insufficient = 20
        self.mock_awi.my_suppliers = ["supplier1"]
        self.agent.negotiators = {"supplier1": MagicMock()}  # Mock negotiator entry

        proposals = self.agent.first_proposals()

        self.assertIn("supplier1", proposals)
        q, t, p = proposals["supplier1"]
        self.assertGreaterEqual(q, 10)  # NMI min for supplier1
        self.assertEqual(q, 20)  # Should propose the full need if within NMI max
        self.assertEqual(p, 5.0)  # NMI min price for supplier1
        self.assertEqual(t, self.mock_awi.current_step + 1)  # Earliest time

    def test_FP_Sales_SingleConsumer_SufficientStock(self):
        self.mock_awi.is_last_level = False
        self.mock_awi.my_consumers = ["consumer1"]
        self.agent.negotiators = {"consumer1": MagicMock()}

        # Simulate available product stock in the agent's IM
        self.agent.im.product_batches.append(
            Batch(str(uuid.uuid4()), 50, 50, 10.0, 0, MaterialType.PRODUCT)
        )
        # And some planned production
        self.agent.im.production_plan = {0: 10, 1: 10}  # Agent is at day 0

        proposals = self.agent.first_proposals()
        self.assertIn("consumer1", proposals)
        q, t, p = proposals["consumer1"]
        # Estimated sellable: 50 (stock) + 10 (day 0 prod) + 10 (day 1 prod) + 10 (day 2 prod) = 80
        # (assuming sellable_horizon = 3, and current_day = 0)
        # qty_per_consumer = 80. NMI max for consumer1 is 30.
        self.assertEqual(q, 30)  # Limited by NMI max of consumer1
        self.assertEqual(p, 25.0)  # NMI max price for consumer1
        self.assertEqual(t, self.mock_awi.current_step + 2)  # Default T+2 for selling

    def test_FP_AgentAtFirstLevel(self):
        self.mock_awi.is_first_level = True
        self.agent.total_insufficient = 20  # Should be ignored
        self.mock_awi.my_suppliers = ["supplier1"]  # Should be ignored for proposals
        self.agent.negotiators = {"supplier1": MagicMock(), "consumer1": MagicMock()}
        self.mock_awi.my_consumers = ["consumer1"]
        self.mock_awi.is_last_level = False
        self.agent.im.product_batches.append(
            Batch(str(uuid.uuid4()), 10, 10, 1.0, 0, MaterialType.PRODUCT)
        )

        proposals = self.agent.first_proposals()
        self.assertNotIn("supplier1", proposals)
        self.assertIn("consumer1", proposals)

    # --- 2.5 _generate_counter_offer Method (Conceptual - depends on score_offers) ---
    # These tests require careful setup of IM, NMI, and offer details.
    # We'll mock score_offers to test the logic of _generate_counter_offer itself.
    @patch.object(LitaAgentCIR, "score_offers")
    def test_GCO_InventoryOpt_Buy_TimeImprovesScore(self, mock_score_offers):
        original_offer = (30, 3, 8.0)  # q, t, p
        partner_id = "supplier1"
        self.agent.negotiator_nmi_cache[partner_id] = self._get_mock_nmi_for_partner(
            partner_id
        )
        self.agent.total_insufficient = 40  # Agent needs more

        # score_offers returns (raw_score, normalized_score)
        # Scenario: orig_t=3. Candidate times: 3, 2 (since t_min for supplier1 is 1, current_day=0)
        # Score for (q_adj, t=3, p_adj_q): e.g., (any, 0.6)
        # Score for (q_adj, t=2, p_adj_qt): e.g., (any, 0.7) -> this should be chosen
        mock_score_offers.side_effect = [
            (100, 0.6),  # Score for original time (t=3) with adjusted Q, P
            (120, 0.7),  # Score for earlier time (t=2) with adjusted Q, P
        ]

        counter = self.agent._generate_counter_offer(
            partner_id,
            original_offer,
            optimize_for_inventory=True,
            optimize_for_profit=False,
        )
        self.assertIsNotNone(counter)
        new_q, new_t, new_p = counter

        # Expected Q: min(orig_q*(1+eps), need_delta=40) = min(33, 40) = 33. Clamped by NMI.
        # NMI for supplier1: q_min=10, q_max=50. So 33 is fine.
        self.assertEqual(new_q, 33)
        self.assertEqual(new_t, 2)  # Time that resulted in better score
        # Price would be orig_p * (1+qty_concession) * (1+time_concession)
        # orig_p=8.0. qty_concession=0.02. time_concession=0.01
        # p_adj_q = 8.0 * 1.02 = 8.16
        # p_adj_qt for t=2 = 8.16 * 1.01 = 8.2416
        self.assertAlmostEqual(new_p, 8.0 * 1.02 * 1.01, places=4)

    @patch.object(LitaAgentCIR, "score_offers")
    def test_GCO_ProfitOpt_PriceChange(self, mock_score_offers):
        original_offer = (20, 3, 18.0)  # q, t, p
        partner_id = "consumer1"  # Selling
        self.agent.negotiator_nmi_cache[partner_id] = self._get_mock_nmi_for_partner(
            partner_id
        )

        # Profit opt doesn't call score_offers in the current _generate_counter_offer
        # It directly adjusts price. Let's remove the mock for this specific test path.
        # Or, ensure optimize_for_inventory=False so time eval part is skipped.

        counter = self.agent._generate_counter_offer(
            partner_id,
            original_offer,
            optimize_for_inventory=False,
            optimize_for_profit=True,
        )
        self.assertIsNotNone(counter)
        new_q, new_t, new_p = counter

        self.assertEqual(new_q, 20)  # Quantity unchanged
        self.assertEqual(new_t, 3)  # Time unchanged
        # Selling: new_p = orig_p * (1 + profit_target_opt=0.05) = 18.0 * 1.05 = 18.9
        # NMI for consumer1: p_min=15, p_max=25. So 18.9 is fine.
        self.assertAlmostEqual(new_p, 18.0 * 1.05, places=4)

    # --- 2.6 counter_all Method (Conceptual - High-level logic flow) ---
    # These tests are the most complex as they integrate many parts.
    # We'd mock _evaluate_offer_combinations and _generate_counter_offer
    # to test the decision tree of counter_all.

    @patch.object(LitaAgentCIR, "_evaluate_offer_combinations")
    @patch.object(LitaAgentCIR, "_generate_counter_offer")
    def test_CA_Case1_AcceptAndCounter(self, mock_gco, mock_eval_comb):
        # Setup for Case 1: norm_score > p_thresh, norm_profit > q_thresh
        # Best combo is one offer, leaves some need.
        best_combo_items = [("supplier1", (20, 2, 7.0))]  # nid, (q,t,p)
        norm_score = 0.8
        norm_profit = 0.1
        mock_eval_comb.return_value = (best_combo_items, norm_score, norm_profit)

        # Mock _generate_counter_offer for the "other" offer
        counter_outcome_for_s2 = (15, 1, 6.5)
        mock_gco.return_value = counter_outcome_for_s2

        # Agent's thresholds
        self.agent.p_threshold = 0.7
        self.agent.q_threshold = 0.0

        # Simulate a remaining need after accepting best_combo
        # This requires mocking self.im.deepcopy().get_total_insufficient_raw()
        # For simplicity, let's assume _generate_counter_offer is called correctly.
        # We need to ensure the agent's IM state reflects a need.
        self.agent.im.production_plan = {1: 50}  # High need
        # Initial stock for supplier1's offer to make sense
        # self.agent.im.raw_material_batches.append(...)

        offers_received = {
            "supplier1": (20, 2, 7.0),  # This one will be in best_combo
            "supplier2": (10, 1, 6.0),  # This one will be countered
        }
        # Mock NMI for supplier2 as well
        self.agent.negotiator_nmi_cache["supplier2"] = create_mock_nmi(
            p_min=5, p_max=10
        )

        responses = self.agent.counter_all(offers_received, states={})

        mock_eval_comb.assert_called_once()

        self.assertEqual(
            responses["supplier1"].response, self.agent.ResponseType.ACCEPT_OFFER
        )
        self.assertEqual(responses["supplier1"].offer, best_combo_items[0][1])

        self.assertEqual(
            responses["supplier2"].response, self.agent.ResponseType.COUNTER_OFFER
        )
        self.assertEqual(responses["supplier2"].offer, counter_outcome_for_s2)

        # Check that _generate_counter_offer was called for supplier2
        # with optimize_for_inventory=True, optimize_for_profit=False, and a target_quantity
        mock_gco.assert_called_once()
        args, kwargs = mock_gco.call_args
        self.assertEqual(args[0], "supplier2")  # nid
        self.assertEqual(args[1], offers_received["supplier2"])  # original_offer
        self.assertTrue(kwargs["optimize_for_inventory"])
        self.assertFalse(kwargs["optimize_for_profit"])
        self.assertIsNotNone(kwargs["inventory_target_quantity"])

    @patch.object(LitaAgentCIR, "_evaluate_offer_combinations")
    @patch.object(LitaAgentCIR, "_generate_counter_offer")
    def test_CA_Case2_CounterInventory(self, mock_gco, mock_eval_comb):
        # Setup for Case 2: norm_score <= p_thresh, norm_profit > q_thresh
        best_combo_items = [("supplier1", (20, 2, 7.0))]
        norm_score = 0.6  # Below p_thresh
        norm_profit = 0.1  # Above q_thresh
        mock_eval_comb.return_value = (best_combo_items, norm_score, norm_profit)

        counter_outcome = (25, 1, 7.5)  # Example counter
        mock_gco.return_value = counter_outcome

        self.agent.p_threshold = 0.7
        self.agent.q_threshold = 0.0

        offers_received = {"supplier1": (20, 2, 7.0)}
        self.agent.negotiator_nmi_cache["supplier1"] = self._get_mock_nmi_for_partner(
            "supplier1"
        )

        responses = self.agent.counter_all(offers_received, states={})

        self.assertEqual(
            responses["supplier1"].response, self.agent.ResponseType.COUNTER_OFFER
        )
        self.assertEqual(responses["supplier1"].offer, counter_outcome)
        mock_gco.assert_called_once_with(
            "supplier1",
            offers_received["supplier1"],
            optimize_for_inventory=True,
            optimize_for_profit=False,  # Case 2: norm_profit > q_thresh
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    # You can add more selective test runs here if needed
    # suite = unittest.TestSuite()
    # suite.addTest(TestLitaAgentCIR("test_FP_Procurement_SingleSupplier"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
