let boardSize = 15;
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const resetBtn = document.getElementById('reset-btn');

let currentBoard = {};
let lastMove = null;
let gameOver = false;
let isThinking = false;

async function loadConfig() {
    try {
        const response = await fetch('/config');
        const data = await response.json();
        boardSize = data.width;
        initBoard();
        resetGame();
    } catch (error) {
        console.error("Error loading config:", error);
        initBoard();
    }
}

function initBoard() {
    boardEl.innerHTML = '';
    boardEl.style.gridTemplateColumns = `repeat(${boardSize}, 40px)`;
    boardEl.style.gridTemplateRows = `repeat(${boardSize}, 40px)`;
    
    // Calculate center marks
    const centers = [];
    if (boardSize % 2 === 1) {
        // Odd: single center
        const mid = Math.floor(boardSize / 2);
        centers.push(`${mid},${mid}`);
    } else {
        // Even: four centers
        const mid2 = boardSize / 2;
        const mid1 = mid2 - 1;
        centers.push(`${mid1},${mid1}`);
        centers.push(`${mid1},${mid2}`);
        centers.push(`${mid2},${mid1}`);
        centers.push(`${mid2},${mid2}`);
    }

    for (let r = 0; r < boardSize; r++) {
        for (let c = 0; c < boardSize; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            
            // Add edge classes for intersection lines
            if (r === 0) cell.classList.add('top-edge');
            if (r === boardSize - 1) cell.classList.add('bottom-edge');
            if (c === 0) cell.classList.add('left-edge');
            if (c === boardSize - 1) cell.classList.add('right-edge');
            
            // Add center mark dot
            if (centers.includes(`${r},${c}`)) {
                const dot = document.createElement('div');
                dot.className = 'center-dot';
                cell.appendChild(dot);
            }

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
    lastMove = move;
    renderBoard(currentBoard); // This will clear previous last-move mark
    renderPiece(move, 1, true); // 1 is Black (Player), true for last move
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

        if (data.ai_move !== undefined) {
            lastMove = data.ai_move;
        } else if (data.status === 'end' && data.last_move !== undefined) {
            lastMove = data.last_move;
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

function renderPiece(move, player, isLastMove = false) {
    const cell = document.querySelector(`.cell[data-move="${move}"]`);
    if (cell) {
        cell.innerHTML = '';
        const piece = document.createElement('div');
        piece.className = `piece ${player === 1 ? 'black' : 'white'}`;
        if (isLastMove) {
            piece.classList.add('last-move');
        }
        cell.appendChild(piece);
    }
}

function renderBoard(states) {
    currentBoard = states;
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
        const move = parseInt(cell.dataset.move);
        const player = states[move];
        cell.innerHTML = '';
        if (player) {
            const piece = document.createElement('div');
            piece.className = `piece ${player === 1 ? 'black' : 'white'}`;
            if (move === lastMove) {
                piece.classList.add('last-move');
            }
            cell.appendChild(piece);
        }
    });
}

function updateStatus(msg) {
    statusEl.textContent = msg;
}

async function resetGame() {
    try {
        await fetch('/reset', { method: 'POST' });
        gameOver = false;
        isThinking = false;
        currentBoard = {};
        lastMove = null;
        initBoard();
        updateStatus("Your Turn (Black)");
    } catch (error) {
        console.error("Error resetting game:", error);
    }
}

resetBtn.addEventListener('click', resetGame);

// Start the game
loadConfig();
