
    # Multi-Core Matrix Compute Unit Subsystem Specification v1.0

    Section 1. Subsystem & Component Overview
    The Matrix Compute Unit (MCU) consists of a 32x32 PE array performing matrix tile calculations.
    Key components include:
    - Matrix ALU: Performs 32x32 MAC operations per cycle
    - Control Unit: FSM-based main controller
    - Register File: 64 vector registers
    - DMA Engine: Transfers matrix tiles between L1 SRAM and DRAM
    - Memory Interface: AXI4-compatible interface

    Section 2. Control & Status Registers
    - CTRL_REG: offset 0x00, access RW, description "MCU Main Control Register"
    - STATUS_REG: offset 0x04, access RO, description "MCU Execution Status Register"

    Section 3. Instruction Set Architecture & Format
    Instruction Format (64 bits):
    - Opcode: 6 bits
    - Destination register: 4 bits
    - Source register 1: 4 bits
    - Source register 2: 4 bits
    - Dimension control: 10 bits
    - Flags: 4 bits

    Instruction Set Table:
    - MLOAD: Opcode 0x01, Load tile into local SRAM buffer
    - MSTORE: Opcode 0x02, Store tile from SRAM buffer to external memory
    - MADD: Opcode 0x03, Execute tile addition
    - MMUL: Opcode 0x04, Execute matrix multiplication

    Section 4. Memory Resources & Buffers
    - SRAM_TILE_BUF: Base address 0x80000000, Size 64KB, SRAM local buffer
    - L1_CACHE: Base address 0x90000000, Size 512KB, L1 Matrix Cache

    Section 5. Data Structures
    Matrix Descriptor format:
    - rows: 16 bits
    - columns: 16 bits
    - format: 8 bits
    - address: 64 bits

    Section 6. Execution Model & Performance
    - Tiling approach for large matrix operations
    - Pipeline execution across 4 matrix stages
    - Frequency: 500 MHz
    - Throughput: 6.25 TFLOPS

    Section 7. Error Handling
    - OVERFLOW_ERR: Triggers when matrix MAC result exceeds 32-bit float range
    