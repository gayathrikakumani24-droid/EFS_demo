"""
Cycle-accurate simulation and verification testbench for Traffic Light Controller.
Verifies:
1. RED = 5 cycles
2. GREEN = 10 cycles
3. YELLOW = 3 cycles
4. Total period = 18 cycles and repeats
5. One-hot output mutual exclusivity at every single cycle
6. Reset behavior from any state (RED, GREEN, YELLOW) forces state to RED and timer_cnt to 0
"""

class TrafficLightController:
    """Cycle-accurate software model of synthesizable SystemVerilog RTL."""
    
    # State encoding
    STATE_RED = 0
    STATE_GREEN = 1
    STATE_YELLOW = 2
    
    STATE_NAMES = {0: "RED", 1: "GREEN", 2: "YELLOW"}
    
    # Durations in cycles
    RED_DURATION = 5
    GREEN_DURATION = 10
    YELLOW_DURATION = 3
    
    def __init__(self):
        self.state = self.STATE_RED
        self.timer_cnt = 0
        self.red_light = 1
        self.green_light = 0
        self.yellow_light = 0
        self.timer_done = 0
        
    def eval_outputs(self):
        """Combinational output decoding from state."""
        self.red_light = 1 if self.state == self.STATE_RED else 0
        self.green_light = 1 if self.state == self.STATE_GREEN else 0
        self.yellow_light = 1 if self.state == self.STATE_YELLOW else 0
        
        # Derived terminal count signal
        if self.state == self.STATE_RED and self.timer_cnt == (self.RED_DURATION - 1):
            self.timer_done = 1
        elif self.state == self.STATE_GREEN and self.timer_cnt == (self.GREEN_DURATION - 1):
            self.timer_done = 1
        elif self.state == self.STATE_YELLOW and self.timer_cnt == (self.YELLOW_DURATION - 1):
            self.timer_done = 1
        else:
            self.timer_done = 0

    def step(self, reset_n: int):
        """Execute one clock cycle (posedge clk)."""
        if reset_n == 0:
            # Asynchronous / synchronous active-low reset
            self.state = self.STATE_RED
            self.timer_cnt = 0
            self.eval_outputs()
            return
            
        # Normal sequential operation
        if self.state == self.STATE_RED:
            if self.timer_cnt == (self.RED_DURATION - 1):
                self.state = self.STATE_GREEN
                self.timer_cnt = 0
            else:
                self.timer_cnt += 1
        elif self.state == self.STATE_GREEN:
            if self.timer_cnt == (self.GREEN_DURATION - 1):
                self.state = self.STATE_YELLOW
                self.timer_cnt = 0
            else:
                self.timer_cnt += 1
        elif self.state == self.STATE_YELLOW:
            if self.timer_cnt == (self.YELLOW_DURATION - 1):
                self.state = self.STATE_RED
                self.timer_cnt = 0
            else:
                self.timer_cnt += 1
                
        self.eval_outputs()


def run_verification():
    dut = TrafficLightController()
    
    # 1. Reset initialization check
    dut.step(reset_n=0)
    assert dut.state == TrafficLightController.STATE_RED, f"Expected RED on reset, got {dut.STATE_NAMES[dut.state]}"
    assert dut.timer_cnt == 0, f"Expected timer_cnt=0 on reset, got {dut.timer_cnt}"
    assert (dut.red_light, dut.green_light, dut.yellow_light) == (1, 0, 0), "Outputs must be RED only on reset"
    assert dut.red_light + dut.green_light + dut.yellow_light == 1, "One-hot mutual exclusivity violated on reset"
    print("[PASS] Assertion 1: Reset forces RED with timer_cnt=0 and one-hot output")

    # 2. Timing verification over 3 full periods (3 * 18 = 54 cycles)
    cycle_history = []
    for cycle in range(1, 55):
        # Sample outputs and state at the beginning of each cycle
        current_state = dut.STATE_NAMES[dut.state]
        cycle_history.append((cycle, current_state, dut.timer_cnt, dut.red_light, dut.green_light, dut.yellow_light))
        
        # Mutual exclusivity assertion: exactly one light high at EVERY cycle
        active_lights = dut.red_light + dut.green_light + dut.yellow_light
        assert active_lights == 1, f"Cycle {cycle}: Mutual exclusivity violated! red={dut.red_light}, green={dut.green_light}, yellow={dut.yellow_light}"
        
        # Advance clock with reset deasserted
        dut.step(reset_n=1)

    # Analyze state durations
    red_cycles = [ch for ch in cycle_history if ch[1] == "RED"]
    green_cycles = [ch for ch in cycle_history if ch[1] == "GREEN"]
    yellow_cycles = [ch for ch in cycle_history if ch[1] == "YELLOW"]
    
    # First full period: cycles 1 to 18
    p1 = cycle_history[:18]
    p1_states = [x[1] for x in p1]
    assert p1_states == ["RED"]*5 + ["GREEN"]*10 + ["YELLOW"]*3, f"Unexpected sequence in period 1: {p1_states}"
    print("[PASS] Assertion 2: Period 1 matches exactly RED (5 cycles) -> GREEN (10 cycles) -> YELLOW (3 cycles)")

    # Second full period: cycles 19 to 36
    p2 = cycle_history[18:36]
    p2_states = [x[1] for x in p2]
    assert p2_states == ["RED"]*5 + ["GREEN"]*10 + ["YELLOW"]*3, f"Unexpected sequence in period 2: {p2_states}"
    print("[PASS] Assertion 3: Period 2 repeats sequence identically (18 cycle wrap-around)")

    # Third full period: cycles 37 to 54
    p3 = cycle_history[36:54]
    p3_states = [x[1] for x in p3]
    assert p3_states == ["RED"]*5 + ["GREEN"]*10 + ["YELLOW"]*3, f"Unexpected sequence in period 3: {p3_states}"
    print("[PASS] Assertion 4: Period 3 repeats sequence identically (total period = 18 cycles)")

    # 3. Reset behavior from any state test
    # Reset while in GREEN
    dut.step(reset_n=1) # advance to RED cycle 1
    for _ in range(5):
        dut.step(reset_n=1) # enter GREEN
    assert dut.state == TrafficLightController.STATE_GREEN
    # Assert reset while in GREEN
    dut.step(reset_n=0)
    assert dut.state == TrafficLightController.STATE_RED
    assert dut.timer_cnt == 0
    assert (dut.red_light, dut.green_light, dut.yellow_light) == (1, 0, 0)
    print("[PASS] Assertion 5: Asynchronous reset from GREEN correctly returns to RED and clears counter")

    # Advance to YELLOW
    dut.step(reset_n=1) # cycle 1
    for _ in range(14):
        dut.step(reset_n=1) # advance to YELLOW
    assert dut.state == TrafficLightController.STATE_YELLOW
    # Assert reset while in YELLOW
    dut.step(reset_n=0)
    assert dut.state == TrafficLightController.STATE_RED
    assert dut.timer_cnt == 0
    assert (dut.red_light, dut.green_light, dut.yellow_light) == (1, 0, 0)
    print("[PASS] Assertion 6: Asynchronous reset from YELLOW correctly returns to RED and clears counter")

    print("\nALL SIMULATION ASSERTIONS PASSED PERFECTLY!")


if __name__ == "__main__":
    run_verification()
