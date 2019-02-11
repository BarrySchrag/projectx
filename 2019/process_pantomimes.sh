#!/bin/bash
# 20190118_151949
# -pix_fmt gray
# ffmpeg  -vf format=gray,format=yuv422p
ffmpeg_path='ffmpeg'
next_t=4
extent_t=10

if [ ! -f  %1-grayscale.mp4 ]; then
	%ffmpeg_path% -i %1.mp4 -vf hue=s=0 %1-grayscale.mp4
fi

if [ ! -e ".\Wave_everyone_a_hello" ]
then
  mkdir Wave_everyone_a_hello
fi
if [ ! -e ".\Wave_everyone_a_hello\%1.mp4" ]
then
  %ffmpeg_path% -i %1.mp4 -ss %next_t% -t %extent_t% .\Wave_everyone_a_hello\%1.mp4
fi

@item=1
@set /a "next_t=%item%*(4+10)+4"	
if [ ! -e ".\Brush_your_teeth" ] 
then
  mkdir Brush_your_teeth
fi
if [ ! -e ".\Brush_your_teeth\%1.mp4" ]
then
  %ffmpeg_path% -i %1.mp4 -ss %next_t% -t %extent_t% .\Brush_your_teeth\%1.mp4
fi

#if not exist ".\Throw_a_football\" mkdir Throw_a_football
#if not exist ".\Wash_your_hands_at_the_sink\" mkdir Wash_your_hands_at_the_sink
#if not exist ".\Put_on_your_coat\" mkdir Put_on_your_coat
#if not exist ".\Wash_your_face_at_the_sink\" mkdir Wash_your_face_at_the_sink
#if not exist ".\Get_a_drink_of_water_from_the_sink\" mkdir Get_a_drink_of_water_from_the_sink
#if not exist ".\Answer_and_talk_on_your_cell_phone\" mkdir Answer_and_talk_on_your_cell_phone
#if not exist ".\Put_on_a_hat\" mkdir Put_on_a_hat 
#if not exist ".\Vaccum_the_living_room\" mkdir Vaccum_the_living_room
#if not exist ".\Run_in_place\" mkdir Run_in_place
#if not exist ".\Saw_wood\" mkdir Saw_wood
#if not exist ".\Play_the_piano\" mkdir Play_the_piano
#if not exist ".\Play_the_guitar\" mkdir Play_the_guitar
#if not exist ".\Go_fishing_cast_your_pole\" mkdir Go_fishing_cast_your_pole
#if not exist ".\Make_a_basketball_hoop_shot\" mkdir Make_a_basketball_hoop_shot
#if not exist ".\Drive_a_golf_ball\" mkdir Drive_a_golf_ball
#if not exist ".\Throw_a_baseball\" mkdir Throw_a_baseball
#if not exist ".\Juggle_three_balls\" mkdir Juggle_three_balls
#if not exist ".\Hit_a_baseball_with_a_bat\" mkdir Hit_a_baseball_with_a_bat
#if not exist ".\Shoot_with_a_bow_and_arrow\" mkdir Shoot_with_a_bow_and_arrow
#if not exist ".\Swim_in_place\" mkdir Swim_in_place
#if not exist ".\Catch_a_baseball_with_a_mit\" mkdir Catch_a_baseball_with_a_mit
#if not exist ".\Wave_goodbye_to_everyone\" mkdir Wave_goodbye_to_everyone