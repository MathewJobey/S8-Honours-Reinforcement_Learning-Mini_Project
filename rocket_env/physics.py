from Box2D.b2 import contactListener

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        # We only have one body part now: The Main Hull.
        # If it touches anything (Ground or Pad), we flag it.
        
        # Note: We don't decide "Crash" vs "Landing" here anymore.
        # We just say "Contact happened". The _compute_reward() function
        # in rocket_lander.py checks the speed/angle to decide if it was a good landing.
        
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True 
            
    def EndContact(self, contact):
        pass