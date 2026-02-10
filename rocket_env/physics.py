from Box2D.b2 import contactListener

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        # Called when two shapes hit each other
        
        # Check if the MAIN BODY hit the ground (Crash)
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True 
            
        # Check if the LEGS hit the ground (Landing)
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    
    def EndContact(self, contact):
        # Called when objects stop touching
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False