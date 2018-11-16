--What was the hottest day in our data set? Where was that?
select zip ,maxtemperaturef from weather where maxtemperaturef=(select max(maxtemperaturef) from weather)
--How many trips started at each station?
select count(trip_id)as trips ,start_terminal from trips group by start_terminal order by start_terminal asc
--What's the shortest trip that happened?
select  trip_id,duration from trips where duration=(select min(duration) from trips)
--What is the average trip duration, by end station?
select end_station,avg(duration) from trips  group by end_station order by end_station