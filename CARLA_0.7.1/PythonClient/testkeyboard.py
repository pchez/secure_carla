import keyboard
import time

def _get_keyboard_control():
    
    #control = VehicleControl()
    print 'looking for key'
    
    if keyboard.is_pressed('a'):
        print 'pressed a'
        #control.steer = -1.0
    if keyboard.is_pressed('d'):
        print 'pressed d' 
        #control.steer = 1.0
    if keyboard.is_pressed('w'):
        print 'pressed w'
        #control.throttle = 1.0
    if keyboard.is_pressed('s'):
        print 'pressed s'
        #control.brake = 1.0
    if keyboard.is_pressed('space'):
        print 'pressed space'
        #control.hand_brake = True
    if keyboard.is_pressed('q'):
        print 'pressed q'
        #self._is_on_reverse = not self._is_on_reverse            
    #control.reverse = self.is_on_reverse
    
    return control

control = None
while True:
    control = _get_keyboard_control()
    time.sleep(0.5)
