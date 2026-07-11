async function runPrediction() {
    const text = document.getElementById('textInput').value.trim();
    const resultBox = document.getElementById('resultBox');

    if (!text) {
        alert('분석할 문장을 입력해주세요.');
        return;
    }

    resultBox.style.display = 'block';
    document.getElementById('inputText').textContent = '로딩 중';
    document.getElementById('basicInfo').textContent = '로딩 중';
    document.getElementById('biasedInfo').textContent = '로딩 중';
    document.getElementById('compareInfo').textContent = '로딩 중';
    document.getElementById('check1').textContent = '로딩 중';
    document.getElementById('check2').textContent = '로딩 중';
    document.getElementById('check3').textContent = '로딩 중';
    document.getElementById('check4').textContent = '로딩 중';
    document.getElementById('check5').textContent = '로딩 중';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();

        document.getElementById('resultBox').style.display = 'block';
        document.getElementById('inputText').textContent = data.text || '-';
        document.getElementById('modelName1').textContent = data.basic.name || '기본(n_model)';
        document.getElementById('basicInfo').textContent = `${data.basic.label} (${(data.basic.confidence * 100).toFixed(2)}%)`;
        document.getElementById('modelName2').textContent = data.biased.name || '편향(p_model)';
        document.getElementById('biasedInfo').textContent = `${data.biased.label} (${(data.biased.confidence * 100).toFixed(2)}%)`;
        document.getElementById('check1').textContent = data.basic.repo_id || 'n_model';
        document.getElementById('check2').textContent = data.biased.repo_id || 'p_model';
        document.getElementById('compareInfo').textContent = data.comparison || '-';
        document.getElementById('check3').textContent = `${data.basic.label} (${(data.basic.confidence * 100).toFixed(2)}%)`;
        document.getElementById('check4').textContent = `${data.biased.label} (${(data.biased.confidence * 100).toFixed(2)}%)`;
        document.getElementById('check5').textContent = data.comparison || '-';
    } catch (err) {
        document.getElementById('resultBox').style.display = 'block';
        document.getElementById('inputText').textContent = '오류';
        document.getElementById('basicInfo').textContent = '오류';
        document.getElementById('biasedInfo').textContent = '오류';
        document.getElementById('compareInfo').textContent = '오류';
        document.getElementById('check3').textContent = '오류';
        document.getElementById('check4').textContent = '오류';
        document.getElementById('check5').textContent = '오류';
        console.error(err);
    }
}
