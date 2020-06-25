console.log("|Hw|");


$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				sentence : $('#sentenceInput').val(),
			},
			type : 'POST',
			url : '/predict'
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(data.name).show();
				$('#errorAlert').hide();
				console.log(data.response["sentence"],data.response["prediction"])
			}

		});

		event.preventDefault();

	});

	$('input[type="submit"]').attr('disabled', true);
    $('textarea').on('keyup', function () {
        var textarea_value = $('input[name="sentenceInput"]').val();
        if (textarea_value != '') {
            $('input[type="submit"]').attr('disabled', false);
        } else {
            $('input[type="submit"]').attr('disabled', true);
        }
    });
});