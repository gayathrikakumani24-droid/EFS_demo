// Self-checking Testbench for sync_ff
module sync_ff_tb;
    parameter int STAGES = 2;
    parameter int WIDTH  = 4;

    logic             clk;
    logic             rst_n;
    logic [WIDTH-1:0] din;
    logic [WIDTH-1:0] dout;

    sync_ff #(
        .STAGES(STAGES),
        .WIDTH(WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .din(din),
        .dout(dout)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        din = 4'b0000;
        #20;
        rst_n = 1;
        #10;
        din = 4'b1010;
        @(posedge clk);
        @(posedge clk);
        #1;
        if (dout !== 4'b1010) begin
            $display("ERROR: sync_ff mismatch. Expected 4'b1010, got %b", dout);
            $finish(1);
        end else begin
            $display("PASS: sync_ff verified successfully.");
            $finish(0);
        end
    end
endmodule
