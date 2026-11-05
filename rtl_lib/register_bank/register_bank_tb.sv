// Self-checking Testbench for register_bank
module register_bank_tb;
    parameter int ADDR_WIDTH = 8;
    parameter int DATA_WIDTH = 32;
    parameter int NUM_REGS   = 4;

    logic                  clk;
    logic                  rst_n;
    logic                  wr_en;
    logic [ADDR_WIDTH-1:0] addr;
    logic [DATA_WIDTH-1:0] wdata;
    logic [3:0]            wstrb;
    logic                  rd_en;
    logic [DATA_WIDTH-1:0] rdata;
    logic [DATA_WIDTH-1:0] reg_out [0:NUM_REGS-1];
    logic [DATA_WIDTH-1:0] reg_in  [0:NUM_REGS-1];

    register_bank #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_REGS(NUM_REGS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .addr(addr),
        .wdata(wdata),
        .wstrb(wstrb),
        .rd_en(rd_en),
        .rdata(rdata),
        .reg_out(reg_out),
        .reg_in(reg_in)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        for (int i = 0; i < NUM_REGS; i++) reg_in[i] = '0;
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        addr  = 0;
        wdata = 0;
        wstrb = 4'hF;
        #20;
        rst_n = 1;
        #10;

        // Write 0xDEADBEEF to reg 1 (addr 4)
        wr_en = 1;
        addr  = 8'h04;
        wdata = 32'hDEADBEEF;
        wstrb = 4'hF;
        @(posedge clk);
        wr_en = 0;
        @(posedge clk);

        // Read back from reg 1
        rd_en = 1;
        addr  = 8'h04;
        @(posedge clk);
        rd_en = 0;
        #1;

        if (rdata !== 32'hDEADBEEF) begin
            $display("ERROR: register_bank mismatch. Expected 32'hDEADBEEF, got %h", rdata);
            $finish(1);
        end else begin
            $display("PASS: register_bank verified successfully.");
            $finish(0);
        end
    end
endmodule
