import os
from pong import Game
from pong.game import GameInformation
import pickle
from sys import argv

import neat
import neat.config # Importing this type for annotaions
import neat.genome # Importing this type for annotaions
import neat.nn.feed_forward # Importing this type for annotaions
import pygame

class PongGame:
    def __init__(self, window: pygame.Surface, width: int, height: int):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome: neat.genome, config: neat.Config):
        """
        Test the performance of a given artificial neural network (from a specific checkpoint)
        by playing with it.
        Human player controls the left paddle and the AI player controls the right paddle.
        """
        pygame.display.set_caption("Pong Game - Play with AI")

        # Create a neural network for the given genome and configurations
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            # Human player controls
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)
            
            # AI player controls (An specific genome from a neat checkpoint, typicaly the last checkpoint) 
            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
            
            # AI Desides to move the right paddle or not based on the current inputs
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else: 
                self.game.move_paddle(left=False, up=True)
            
            self.game.loop()
            self.game.draw(draw_score=True, draw_hits=False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1: neat.genome, genome2: neat.genome, config: neat.Config) -> None:
        """
        Finds the fitness factor by performing
        a simulation game on the two given genomes
        playing against each other.
        """
        pygame.display.set_caption("Pong Game - Training the AI")

        # Creating the neural nets
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config) 

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1))
            
            # Deside to move the left paddle or not based on the output
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else: 
                self.game.move_paddle(left=True, up=True)
            
            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))
            
            # Deside to move the right paddle or not based on the output
            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else: 
                self.game.move_paddle(left=False, up=True)

            game_info = self.game.loop()
            
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1: neat.genome, genome2: neat.genome, game_info: GameInformation) -> None:
        """
        Calculates the fitness of both genomes by summing the
        current values of the with the new given number of hits.
        """
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


def eval_genomes(genomes: list[tuple], config: neat.Config) -> None:
    """
    Takes all of the genomes in the current generion
    and sets a fitness for each one of them.
    """

    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i < len(genomes) - 1:
            genome1.fitness = 0
            for genome_id2, genome2 in genomes[i+1:]:
                genome2.fitness = 0 if genome2.fitness == None else genome2.fitness

                game = PongGame(window, width, height)
                game.train_ai(genome1, genome2, config)
        else:
            break


def run_neat(config: neat.Config, checkpoint=None) -> None:
    """
    Runs the neat algorithm from the beginning or the given chekpoint,
    adjusting to the given configuration, along with some reporters.
    """
    
    # Check if any checkpoint has given 
    if checkpoint:
        p = neat.Checkpointer.restore_checkpoint(checkpoint) #* Run from a specific checkpoint.
    else:
        p = neat.Population(config)
    # Adding some reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes, 10) # Gives the best neural network
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_best_ai(config: neat.config):
    """Testing the the best genome which is stored on the 'best.pickle' file."""

    # Seting up a game
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    # Taking the best AI (genome) from our pickle file
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    
    # Initializing a game
    game = PongGame(window, width, height)
    game.test_ai(winner, config)


def main() -> None:
    # Finding the file's path
    local_dir = os.path.dirname(__file__)
    # Stroing the path of the config file
    config_path = os.path.join(local_dir, "config.txt")
    
    # Passing different properties form the configuration file that we want to use
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        if argv[1].lower().strip() == 'train':
            # Cheack if user passes second argument for training which must be a checkpoint
            if len(argv) == 3:
                checkpoint = argv[2]
            else:
                checkpoint = None 

            # Runing the Neat algorithm
            run_neat(config, checkpoint)
        
        elif argv[1].lower().strip() == 'test':
            # Testing the AI
            test_best_ai(config)
        
        else:
            raise Exception("WrongCommand: You Need to Give 'train' or 'test' as the first argument.")
    
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()    
