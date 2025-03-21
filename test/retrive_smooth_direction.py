'''
Script to make eye traker move smoothly from one cell to another
without jumping when testing liven the CNN model
'''

grid = [
    [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
]
GRID_COLS = 10
GRID_ROWS = 10

def euclidian_distance_between_cells(cell1, cell2):
    row1, col1 = (cell1 - 1) // GRID_COLS, (cell1 - 1) % GRID_COLS
    row2, col2 = (cell2 - 1) // GRID_COLS, (cell2 - 1) % GRID_COLS
    return ((row2 - row1) ** 2 + (col2 - col1) ** 2) ** 0.5
def get_possible_cells(current):
    # return possible cells to move to from current cell (up, down, left, right, up-left, up-right, down-left, down-right)
    row, col = (current - 1) // GRID_COLS, (current - 1) % GRID_COLS
    possible_cells = []
    if row > 0:
        possible_cells.append(grid[row - 1][col])
    if row < GRID_ROWS - 1:
        possible_cells.append(grid[row + 1][col])
    if col > 0:
        possible_cells.append(grid[row][col - 1])
    if col < GRID_COLS - 1:
        possible_cells.append(grid[row][col + 1])
    if row > 0 and col > 0:
        possible_cells.append(grid[row - 1][col - 1])
    if row > 0 and col < GRID_COLS - 1:
        possible_cells.append(grid[row - 1][col + 1])
    if row < GRID_ROWS - 1 and col > 0:
        possible_cells.append(grid[row + 1][col - 1])
    if row < GRID_ROWS - 1 and col < GRID_COLS - 1:
        possible_cells.append(grid[row + 1][col + 1])
    return possible_cells
def next_cell(current, desire):
    if current == desire:
        return current
    possible_cells = get_possible_cells(current)
    closest_cell = None
    min_distance = float('inf')
    for cell in possible_cells:
        distance = euclidian_distance_between_cells(cell, desire)
        if distance < min_distance:
            min_distance = distance
            closest_cell = cell
    return closest_cell
def pritty_print(grid):
    for row in grid:
        print(row)
    print()
'''
current_cell = 43
while True:
    desire_cell = int(input('Enter desire cell: '))
    print(f"from {current_cell} to {desire_cell}. End up in:")
    current_cell = next_cell(current_cell, desire_cell)
    print(current_cell)
    pritty_print(grid)
    print()
'''