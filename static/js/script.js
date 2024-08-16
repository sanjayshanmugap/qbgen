document.addEventListener('DOMContentLoaded', function() {
    const generateCluesButton = document.getElementById('generateClues');
    const answerlineInput = document.getElementById('answerline');
    const categoriesSelect = $('#categories');
    const difficultiesSelect = $('#difficulties');
    const similarityThresholdInput = document.getElementById('similarityThreshold');
    const thresholdValueDisplay = document.getElementById('thresholdValue');
    const cluesList = document.getElementById('cluesList');
    const cluesCount = document.getElementById('cluesCount');
    const exportCardsButton = document.getElementById('exportCards');

    $('.select2').select2();

    similarityThresholdInput.addEventListener('input', function() {
        thresholdValueDisplay.textContent = similarityThresholdInput.value;
    });

    generateCluesButton.addEventListener('click', function() {
        const answerline = answerlineInput.value;
        const categories = categoriesSelect.val().join(',');
        const difficulties = difficultiesSelect.val().join(',');
        const similarityThreshold = similarityThresholdInput.value;

        fetch('/process_clues', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                answer: answerline,
                categories: categories,
                difficulties: difficulties,
                similarity_threshold: similarityThreshold
            }),
        })
        .then(response => response.json())
        .then(data => {
            cluesList.innerHTML = '';
            cluesCount.innerHTML = `Number of unique clues: ${data.length}`;
            data.forEach(clue => {
                const li = document.createElement('li');
                li.textContent = clue;
                cluesList.appendChild(li);
            });

            exportCardsButton.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    document.getElementById('exportCards').addEventListener('click', function() {
        const clues = Array.from(document.querySelectorAll('#cluesList li')).map(li => li.textContent);
        const answerline = document.getElementById('answerline').value;

        fetch('/generate_apkg', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ clues: clues, answerline: answerline }),
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(new Blob([blob]));
            const a = document.createElement('a');
            a.href = url;
            a.download = `${answerline}_cards.apkg`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        });
    });
});
