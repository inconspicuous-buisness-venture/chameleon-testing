const submitButton = document.querySelector(".bt-act");


document.addEventListener("DOMContentLoaded", function() {
    const sliders = document.querySelectorAll(".slider");
    const valueSpans = document.querySelectorAll(".settings span");

    let thresholdValue = 15;
    let temperatureValue = 0.95;
    let iterationsValue = 5;
    let durationValue = 120;

    function updateLabelAndValue(index) {
        return function(event) {
            if (index === 1) { 
                temperatureValue = parseInt(event.target.value) / 100;
                valueSpans[index].textContent = temperatureValue; 
            } else {
                valueSpans[index].textContent = event.target.value; 
            }

            switch (index) {
                case 0:
                    thresholdValue = parseInt(event.target.value);
                    break;
                case 2:
                    iterationsValue = parseInt(event.target.value);
                    break;
                case 3:
                    durationValue = parseInt(event.target.value);
                    break;
                default:
                    break;
            }
        };
    }

    sliders.forEach(function(slider, index) {
        slider.addEventListener("input", updateLabelAndValue(index));
    });

});



function submitText() {
    const thresholdValue = document.querySelector("#thre").value;
    const temperatureValue = document.querySelector("#temp").value / 100;
    const iterationsValue = document.querySelector("#iter").value;
    const durationValue = document.querySelector("#dura").value;

    let iterationCount = 0;

    function submit() {
        
        if (iterationCount < iterationsValue) {
            document.getElementById("iterval2").innerHTML = iterationCount;
            document.getElementById("loading").style.display = "inline-block";
            fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    thresholdValue: thresholdValue,
                    temperatureValue: temperatureValue,
                    iterationsValue: iterationsValue,
                    durationValue: durationValue,
                    text: document.getElementById("text").innerHTML.slice(25),
                    prompt: document.getElementById("prompt").innerHTML.slice(25),
                    model: document.getElementById("modelSelect").options[document.getElementById("modelSelect").selectedIndex].value,
                    algorithm: document.getElementById("algorithmSelect").options[document.getElementById("algorithmSelect").selectedIndex].value
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.status == "success") {
                    document.getElementById("text").innerHTML = data.output;
                    document.getElementById("loading").style.display = "none";
                    iterationCount++;
                    submit();
                } else {
                    document.getElementById("loading").style.display = "none";
                    iterationCount++;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById("loading").style.display = "none";
            });
        }
    }

    submit();
}
