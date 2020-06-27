console.log("|Hw|");


$(document).ready(function() {

	$('form').on('submit', function (event) {
		var me = $(this);
		event.preventDefault();
		
		if ( me.data('requestRunning') ) {
			return;
		}
		
		me.data('requestRunning', true);
		
		$.ajax({
			data : {
				sentence : $('#sentenceInput').val(),
			},
			type : 'POST',
			url: '/predict',
			complete: function() {
				me.data('requestRunning', false);
			}
		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(data.name).show();
				$('#result').text(data.response["prediction"])
				$('#errorAlert').hide();
				console.log(data.response["sentence"],data.response["prediction"])
			}

		});

		
		
	});

	$('button[type="submit"]').attr('disabled', true);
    $('textarea').on('keyup', function () {
        var textarea_value = $('input[name="sentenceInput"]').val();
        if (textarea_value != '') {
            $('button[type="submit"]').attr('disabled', false);
        } else {
            $('button[type="submit"]').attr('disabled', true);
        }
    });
});