from statemachine import State
from statemachine import StateMachine

class GripperStateMachine(StateMachine):

    ### States
    EntryState = State(name = 'EntryState', initial = True)

    # Level 1
    State0 = State(name = 'State0', value = 0, enter = "State0Action", exit = "Stop")

    State1 = State(name = 'State1', value = 1, enter = "Link1Rotates", exit = "Stop")

    # Level 2
    State2 = State(name = 'State2', value = 2, enter = "Link2Rotates", exit = "Stop")
    
    State3 = State(name = 'State3', value = 3, enter = "Link3Rotates", exit = "Stop")

    State4 = State(name = 'State4', value = 4, final = True, enter = "Stop")

    State5 = State(name = 'State5', value = 5, enter = "Link2Rotates", exit = "Stop")

    # Level 3
    State6 = State(name = 'State6', value = 6, enter = "Link3Rotates", exit = "Stop")

    State7 = State(name = 'State7', value = 7, final = True, enter = "Stop")

    State8 = State(name = 'State8', value = 8, enter = "Link3Rotates", exit = "Stop")

    State9 = State(name = 'State9', value = 9, final = True, enter = "Stop")

    State10 = State(name = 'State10', value = 10, final = True, enter = "Stop")

    State11 = State(name = 'State11', value = 11, enter = "Link3Rotates", exit = "Stop")

    State12 = State(name = 'State12', value = 12, final = True, enter = "Stop")

    State13 = State(name = 'State13', value = 13, final = True, enter = "Stop")
 
    # Level 4
    State14 = State(name = 'State14', value = 14, final = True, enter = "Stop")

    State15 = State(name = 'State15', value = 15, final = True, enter = "Stop")

    State16 = State(name = 'State16', value = 16, final = True, enter = "Stop")

    State17 = State(name = 'State17', value = 17, final = True, enter = "Stop")

    State18 = State(name = 'State18', value = 18, final = True, enter = "Stop")

    State19 = State(name = 'State19', value = 19, final = True, enter = "Stop")

    ### Transitions Event = [C1 C2 C3 L1 L2 L3 L4],
    # Ci = 1 indicates that link i is blocked by the object
    # Li = 1 indicates that link i has reached the upper limit 
    # L4 = 1 indicates that link3 has reached the lower limit

    Close = EntryState.to(State0)
    
    Event1 = State0.to(State2) | State1.to(State2)

    Event2 = State0.to(State3) | State1.to(State3) | State5.to(State11) | State2.to(State8)

    Event3 = (
        State1.to(State4) | State0.to(State4) | State2.to(State7) | State3.to(State10) |
        State5.to(State12) | State6.to(State14) | State8.to(State16) | State11.to(State18)
    )

    Event4 = State1.to(State5)

    Event5 = State5.to(State13) | State2.to(State6)

    Event6 = (
        State6.to(State15) | State8.to(State17) |
        State3.to(State9) | State11.to(State19)
    )

    Event7 = State0.to(State1)

    # Entering/Exiting action for states

    def __init__(self, finger):
        self.finger = finger
        self.action = "0000"
        super().__init__()

    def Stop(self):
        self.action = "0000"

    def State0Action(self):
        self.action = "1001"

    def Link1Rotates(self):
        self.action = "1000"

    def Link2Rotates(self):
        self.action = "0100"
        
    def Link3Rotates(self):
        self.action = "0010"