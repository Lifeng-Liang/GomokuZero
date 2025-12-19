const boardSize = 8;
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const resetBtn = document.getElementById('reset-btn');

let currentBoard = {};
let gameOver = false;
let isThinking = false;

function initBoard() {
    boardEl.innerHTML = '';
    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            const move = r * boardSize + c;
            cell.dataset.move = move;
            cell.addEventListener('click', () => handleCellClick(move));
            boardEl.appendChild(cell);
        }
    }
}

async function handleCellClick(move) {
    if (gameOver || isThinking || currentBoard[move]) return;

    // Optimistic update: show player's piece immediately
    isThinking = true;
    renderPiece(move, 1); // 1 is Black (Player)
    currentBoard[move] = 1;
    updateStatus("AI is thinking...");

    try {
        const response = await fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: move })
        });

        const data = await response.json();
        isThinking = false;

        if (data.status === 'error') {
            updateStatus(data.message);
            isThinking = false;
            // The piece was added optimistically, so we should re-render from the currentBoard state to "remove" it if the server rejected it.
            // But currentBoard was also updated, so we need to revert that too.
            delete currentBoard[move];
            renderBoard(currentBoard);
            return;
        }

        renderBoard(data.board);

        if (data.status === 'end') {
            gameOver = true;
            if (data.winner === -1) {
                updateStatus("Game Over: Tie!");
            } else {
                updateStatus(`Game Over: ${data.winner === 1 ? 'Black' : 'White'} Wins!`);
            }
        } else {
            updateStatus("Your Turn (Black)");
        }
    } catch (error) {
        isThinking = false;
        console.error("Error making move:", error);
        updateStatus("Error communicating with server.");
    }
}

function renderPiece(move, player) {
    const cell = document.querySelector(`.cell[data-move="${move}"]`);
    if (cell) {
        cell.innerHTML = '';
        const piece = document.createElement('div');
        piece.className = `piece ${player === 1 ? 'black' : 'white'}`;
        cell.appendChild(piece);
    }
}

function renderBoard(states) {
    currentBoard = states;
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
        const move = cell.dataset.move;
        const player = states[move];
        cell.innerHTML = '';
        if (player) {
            const piece = document.createElement('div');
            piece.className = `piece ${player === 1 ? 'black' : 'white'}`;
            cell.appendChild(piece);
        }
    });
}

function updateStatus(msg) {
    statusEl.textContent = msg;
}

resetBtn.addEventListener('click', async () => {
    try {
        await fetch('/reset', { method: 'POST' });
        gameOver = false;
        isThinking = false;
        currentBoard = {};
        initBoard();
        updateStatus("Your Turn (Black)");
    } catch (error) {
        console.error("Error resetting game:", error);
    }
});

// Start the game
initBoard();
