document.getElementById('run-tests-button').addEventListener('click', function() {
    fetch('/run_tests', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(data => {
        console.log(data);
        // Aquí puedes mostrar los resultados en la interfaz de usuario
    });
});