from Box2D.b2 import contactListener

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        # 1. Grab the name tags of whatever just collided. 
        # (The ground will just say 'None')
        tag_a = contact.fixtureA.body.userData
        tag_b = contact.fixtureB.body.userData
        
        # 2. If either of those tags says "rocket", trigger the game over!
        if tag_a == "rocket" or tag_b == "rocket":
            self.env.game_over = True 
            
    def EndContact(self, contact):
        pass