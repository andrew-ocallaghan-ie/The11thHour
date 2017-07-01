<!DOCTYPE html>

<html>
    
    <head>
        
        <title>Dublin Bus Predictions</title>
        
        <!-- Add icon library -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

        <!--Google Fonts-->
        <link href="https://fonts.googleapis.com/css?family=Open+Sans:300" rel="stylesheet">
        
        <!--Responsive Design-->
        <meta name="viewport" content="width=device-width, initial-scale=1">
        
        <!--Encoding-->
        <meta charset="utf-8">

        <!--Bootstrap CSS Links-->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        
        <!--Our Stylesheet, goes at bottom of CSS sheets to ensure precedence-->
        <link href='../static/style.css' rel='stylesheet' type="text/css">
        
        <!--Google charts API-->
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        
        <!--ChartJS-->
        <script src="../static/chartjs/node_modules/chart.js/dist/Chart.js"></script>

        <!--Our functions-->
        <script type="text/javascript">
            
        // Google Map
        function myMap() {
            $.getJSON('http://localhost:5000/api/routes/42', function(json) {
                // Looping through elements in JSON file.
                // Set at route 42 for now. This will be changeable
                // Make empty marker array to put the markers for the marker clustering
                var markers = [];
                for (var i = 0; i < json.stops.length; i++) {
                    // Creating Marker
                    var marker = new google.maps.Marker({
                        position: {lat: parseFloat(json.stops[i].latitude), lng: parseFloat(json.stops[i].longitude)},
                        icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            strokeColor: '#4682b4',
                            strokeWeight: 1.5,
                            fillColor: '#4682b4',
                            fillOpacity: 0.4,
                            scale: 8
                        },
                        map: map});
                    
                    // Push the marker into the marker array for clustering
                    markers.push(marker);
                    // Create infowindow on mouseover
                    var infowindow = new google.maps.InfoWindow();
                    // Contents of infowindow
                    var html = '<h4>' + json.stops[i].shortname + '</h4>';
                    marker.html = html;
                    // Name of station
                    marker.name = json.stops[i].shortname;
                    // Mouse Over Functionality
                    google.maps.event.addListener(marker,'mouseover',function() {
                        // Set contents
                        infowindow.setContent(this.html);
                        // Open infowindow
                        infowindow.open(map, this);
                    });
                }
                
                // Add a marker clusterer to manage the markers.
                var markerCluster = new MarkerClusterer(map, markers,
                    {imagePath: '../static/marker_clustering/images/m'});
            });
            
            
            //dublin bike
             $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};       jQuery.getJSON($SCRIPT_ROOT+"/all",null,function(data){
           //window.alert(JSON.stringify(data));
            var stations = data.stations;
           //window.alert(JSON.stringify(stations));
            _.forEach(stations,function(station){
                var marker = new google.maps.Marker({
                    position:{
                        lat:station.Position_lat,
                        lng:station.Position_lng
                    },
                    icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            strokeColor:'#123456',
                            strokeWeight: 1.5,
                            fillColor: '#123456',
                            fillOpacity: 0.4,
                            scale: 8
                        },
                    map:map,
                    title:station.Name,
                    station_number:station.Number
                });
                
            
                infowindow = new google.maps.InfoWindow();
                
                
                marker.addListener("click",function(){
                 
                
                var contentString='<h3>'+station.Name+'</h3>'
                infowindow.setContent(contentString);
                infowindow.open(map, this);
                }); 
                        
                 
            })
            
    });
            
            
            //Dart
     jQuery.getJSON($SCRIPT_ROOT+"/dart",null,function(data){
           //window.alert(JSON.stringify(data));
            var stations = data.stops;
           //window.alert(JSON.stringify(stations));
            _.forEach(stations,function(station){
                var marker = new google.maps.Marker({
                    position:{
                        lat:station.latitude,
                        lng:station.longitude
                    },
                    icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            strokeColor:'#0000FF',
                            strokeWeight: 1.5,
                            fillColor: '#0000FF',
                            fillOpacity: 0.4,
                            scale: 8
                        },
                    map:map,
                    title:station.Name,
                    station_number:station.Number
                });
                
            
                infowindow = new google.maps.InfoWindow();
                
                
                marker.addListener("click",function(){
                 
                
                var contentString='<h3>'+station.name+'</h3>'
                infowindow.setContent(contentString);
                infowindow.open(map, this);
                }); 
                        
                 
            })
            
    });
            
            
            //Luas
      jQuery.getJSON($SCRIPT_ROOT+"/luas",null,function(data){
           //window.alert(JSON.stringify(data));
            var stations = data.stops;
           //window.alert(JSON.stringify(stations));
            _.forEach(stations,function(station){
                var marker = new google.maps.Marker({
                    position:{
                        lat:station.latitude,
                        lng:station.longitude
                    },
                    icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            strokeColor:'#808000',
                            strokeWeight: 1.5,
                            fillColor: '#808000',
                            fillOpacity: 0.4,
                            scale: 8
                        },
                    map:map,
                    title:station.Name,
                    station_number:station.Number
                });
                
            
                infowindow = new google.maps.InfoWindow();
                
                
                marker.addListener("click",function(){
                 
                
                var contentString='<h3>'+station.name+'</h3>'
                infowindow.setContent(contentString);
                infowindow.open(map, this);
                }); 
                        
                 
            })
            
    });
            
            
            
            // Centre of map
            var myCenter = new google.maps.LatLng(53.3498,-6.2603);
            // Initializing canvas
            var mapCanvas = document.getElementById("map");
            // Map options
            var mapOptions = {
                center: myCenter, 
                zoom: 13
            };
            // Creating map
            var map = new google.maps.Map(mapCanvas, mapOptions);
        }
        </script>
    
    </head>
    
    <body>
        <nav class="navbar navbar-default">
          <div class="container-fluid">
            <div class="navbar-header">
              <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
              <a class="navbar-brand" href="#">Dublin Bus</a>
            </div>

            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
              <form method="post" class="navbar-form navbar-left">
                <div class="form-group">
                  <input type="text" class="form-control" id="origin" name="origin" placeholder="Start">
                  <input type="text" class="form-control" id="destination" name="destination" placeholder="End">
        Dublin Bike: <input type="checkbox" id="db" onclick="boxclick(this.'bike')" /> &nbsp;&nbsp;
      Dart: <input type="checkbox" id="dart"  onclick="boxclick(this.'dart')" /> &nbsp;&nbsp;
      Luas: <input type="checkbox" id="Luas" onclick="boxclick(this.'luas')" />
                </div>
                  <div class="form-group">
                      <select class="form-control" id="now_arrive_depart">
                            <option>Now</option>
                            <option>Arrive By</option>
                            <option>Leave At</option>
                      </select>
                </div>
                <button type="submit" class="btn btn-default btn-circle">GO!</button>
              </form>
              <ul class="nav navbar-nav navbar-right">
                <li><img class="img-responsive" id="weather" src="../static/images/flurries.png"/></li>

              </ul>
            </div>
          </div>
        </nav>
        
        <div class="container">
            <div class="row">
                    <div class="col-sm-12 col-lg-6" id="map"></div>
                    <div class="col-sm-6 col-lg-3" id="bus-info">
                        <h2>Bus Info</h2>
                        <ul>
                            <li>42a - 2.15pm</li>
                            <li>42 - 2.18pm</li>
                            <li>54 - 2.13pm</li>
                            <li>177 - 3.09pm</li>
                            <li>42a - 3.15pm</li>
                        </ul>
                    </div>
                    <!-- Chart -->
                <div class="col-sm-6 col-lg-3">
                    <canvas id="busChart"></canvas>
                </div>

            </div>
 
        </div>

        <!--Script for the chart. Currently uses manual input-->
        <!--Will be cleaner once we have data going in from API-->
        <!--Has to go at the bottom-->


        <script>
            var ctx = document.getElementById("busChart");
            var chartDemo = new Chart(ctx, {
             type: 'line',
             data: {
             labels: ["09:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00"],
             datasets: [{
                 fill: true,
                 label: '42a',
                 data: [48,23,32,55,13,33,22,15],
                 backgroundColor: 'rgba(255, 99, 132, 0.2)',
                 borderColor:'rgba(255,99,132,1)',
                 borderWidth: 1
                        },
                       {
                fill: true,
                label: '37',
                data: [12,13,22,8,43,33,23,17],
                backgroundColor: 'rgba(0, 102, 255, 0.2)',
                borderColor:'rgba(0, 102, 255,1)',
                borderWidth: 1}]
                    },
             options: {
                 maintainAspectRatio: false,
                 responsive: true,
                 title: {
                    fontSize: 15,
                    fontFamily: "'Open Sans', 'Helvetica', 'Arial', sans-serif",
                    display: true,
                    text: "Your Route Travel Times"
                 }
             }
            });
        </script>

        <!--import Lodash-->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/2.4.1/lodash.min.js"></script>
        
        <!--import Jquery-->
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
    
        <!--Google Maps API script-->
        <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAkUZsgbKCDZNWjntnPv5mQJplie2G4h64&callback=myMap"></script>
            
        <!--Google Maps marker clustering script-->
        <script src="../static/marker_clustering/markerclusterer.js"></script>
    
    </body>

</html>