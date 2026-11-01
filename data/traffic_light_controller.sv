// ==============================================================================
// Module: traffic_light_controller
// Description: Synthesizable Traffic Light Controller FSM with internal cycle counter
// Specifications:
//   - Inputs: clk, reset_n (active-low asynchronous reset)
//   - Outputs: red_light, green_light, yellow_light, timer_done
//   - Timing:
//       RED lasts 5 cycles -> GREEN lasts 10 cycles -> YELLOW lasts 3 cycles -> RED
//   - Sequence repeats periodically (total period = 18 cycles)
//   - Mutual exclusivity: Exactly one light active high at any time
// ==============================================================================

module traffic_light_controller (
    input  logic clk,
    input  logic reset_n,
    output logic red_light,
    output logic green_light,
    output logic yellow_light,
    output logic timer_done
);

    // State definitions
    typedef enum logic [1:0] {
        STATE_RED    = 2'b00,
        STATE_GREEN  = 2'b01,
        STATE_YELLOW = 2'b10
    } state_t;

    state_t state_reg, state_next;

    // Cycle duration constants derived from EFS IR duration_cycles
    localparam int RED_CYCLES    = 5;
    localparam int GREEN_CYCLES  = 10;
    localparam int YELLOW_CYCLES = 3;

    // 4-bit cycle counter register (sized to hold max duration 10 cycles)
    logic [3:0] timer_cnt_reg, timer_cnt_next;

    // Sequential process (asynchronous active-low reset to RED with timer_cnt cleared)
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state_reg     <= STATE_RED;
            timer_cnt_reg <= 4'd0;
        end else begin
            state_reg     <= state_next;
            timer_cnt_reg <= timer_cnt_next;
        end
    end

    // Combinational next-state & timer progression logic
    always_comb begin
        state_next     = state_reg;
        timer_cnt_next = timer_cnt_reg + 4'd1;

        case (state_reg)
            STATE_RED: begin
                // Occupied for exactly 5 clock cycles: timer_cnt = 0, 1, 2, 3, 4
                if (timer_cnt_reg == (RED_CYCLES - 1)) begin
                    state_next     = STATE_GREEN;
                    timer_cnt_next = 4'd0;
                end
            end

            STATE_GREEN: begin
                // Occupied for exactly 10 clock cycles: timer_cnt = 0, 1, ..., 9
                if (timer_cnt_reg == (GREEN_CYCLES - 1)) begin
                    state_next     = STATE_YELLOW;
                    timer_cnt_next = 4'd0;
                end
            end

            STATE_YELLOW: begin
                // Occupied for exactly 3 clock cycles: timer_cnt = 0, 1, 2
                if (timer_cnt_reg == (YELLOW_CYCLES - 1)) begin
                    state_next     = STATE_RED;
                    timer_cnt_next = 4'd0;
                end
            end

            default: begin
                state_next     = STATE_RED;
                timer_cnt_next = 4'd0;
            end
        endcase
    end

    // Output logic: strict one-hot mutual exclusivity in every state
    always_comb begin
        red_light    = (state_reg == STATE_RED);
        green_light  = (state_reg == STATE_GREEN);
        yellow_light = (state_reg == STATE_YELLOW);

        // Terminal count indicator for currently active state
        timer_done   = ((state_reg == STATE_RED)    && (timer_cnt_reg == (RED_CYCLES - 1)))    ||
                       ((state_reg == STATE_GREEN)  && (timer_cnt_reg == (GREEN_CYCLES - 1)))  ||
                       ((state_reg == STATE_YELLOW) && (timer_cnt_reg == (YELLOW_CYCLES - 1)));
    end

endmodule
