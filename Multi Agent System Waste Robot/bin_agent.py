import spade
import random
import asyncio
from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour, CyclicBehaviour, OneShotBehaviour
from spade.message import Message
import ast



class BinAgent(spade.agent.Agent):
    def __init__(self, jid, password, position, environment): 
        super().__init__(jid, password)
        self.position = position
        self.environment = environment
        self.max_capacity = 100
        self.current_waste = random.randint(0, int(self.max_capacity * 0.4))
        self.received_responses = {}
        self.accumulation_period = 2
        self.is_waiting_for_truck = False
        self.sent_collection_request = False
        self.resolving = 0
        self.collection_time = []   # Stores the time it took to be collected 
        self.waste = None
        self.time = None
        self.counter = 0

    class WasteAccumulationBehaviour(PeriodicBehaviour):
        async def run(self):
            accumulation = random.randint(1,10)
            if self.agent.max_capacity - self.agent.current_waste < accumulation:
                self.agent.current_waste = 100
                print(f"[{self.agent.name}] [({self.agent.position[1]},{self.agent.position[0]})] is full. Waiting for collection") # We need to decide what happens with bins that are already full... Do we give them more priority??
                if self.agent.sent_collection_request == False and self.agent.is_waiting_for_truck == False:
                    self.agent.sent_collection_request = True
                    self.agent.received_responses = {}
                    await self.send_cfp_to_trucks()
                return

            self.agent.current_waste += accumulation
            print(f"[{self.agent.name}] Waste level : {self.agent.current_waste} / {self.agent.max_capacity}. Accumulation = {accumulation}")

            if self.agent.current_waste < 70:
                counter = 0

            if self.agent.current_waste >= 0.7 * self.agent.max_capacity:
                self.agent.counter += 1
                if self.agent.sent_collection_request == False and self.agent.is_waiting_for_truck == False:
                    print(f"[{self.agent.name}] [({self.agent.position[1]},{self.agent.position[0]})] Waste level reached >= 70%. Sending collection request")
                    self.agent.sent_collection_request = True
                    self.agent.received_responses = {}
                    await self.send_cfp_to_trucks()
            

    
        async def send_cfp_to_trucks(self):
            # Send CFP (Call for Proposals) to all trucks registered in the environment
            print(f"[{self.agent.name}] Attempting to send CFP")
            for truck in self.agent.environment.trucks:
                if not truck.is_busy:
                    cfp_message = Message(to = str(truck.jid))
                    cfp_message.set_metadata("performative", "cfp")
                    cfp_message.body = f"{self.agent.position[0]},{self.agent.position[1]}"
                    await self.send(cfp_message)
                    print(f"[{self.agent.name}] [({self.agent.position[1]},{self.agent.position[0]})] CFP sent to truck {truck.jid}.")
        
            # Add a waiting behavior to collect responses from trucks
            wait_for_responses=self.agent.WaitForResponsesBehaviour()
            self.agent.add_behaviour(wait_for_responses)

    class ReceiveProposalBehaviour(CyclicBehaviour):
        async def run(self):
            # Receive proposals or refusals from trucks
            msg = await self.receive(timeout=1)
            if msg:
                if msg.metadata.get("performative") == "propose":
                    # Extract path, cost and available capacity from truck proposal
                    path_str, estimated_cost, available_capacity, fuel = msg.body.split(";")
                    estimated_cost = int(estimated_cost)
                    available_capacity = int(available_capacity)
                    # Store the proposal, including the path
                    self.agent.received_responses[msg.sender] = {
                        'type': 'proposal',
                        'cost': estimated_cost,
                        'available_capacity': available_capacity,
                        'fuel': fuel,
                        'path': path_str
                    }
                    print(f"[{self.agent.name}] Proposal received from {msg.sender}: Capacity {available_capacity}, Cost {estimated_cost}")
                elif msg.metadata.get("performative") == "decline":
                    # Store the refusal as response
                    self.agent.received_responses[msg.sender] = {'type': 'decline'}
                    print(f"[{self.agent.name}] Rejection received from {msg.sender}")

    class WaitForResponsesBehaviour(OneShotBehaviour):
        async def run(self):
            await asyncio.sleep(3) # wait for trucks responses to be sent
            # End waiting when time expires and evaluate proposals
            print(f"[{self.agent.name}] Wait time expired. Evaluating proposals...")
            await self.evaluate_proposals()
            self.agent.received_responses = {}
            self.kill()  # End behavior after evaluation

        async def evaluate_proposals(self):
            # Select the best proposal among valid responses (not decline)
            best_proposal = None
            for truck_jid, response in self.agent.received_responses.items():
                if response['type'] == 'proposal':
                    # get the right truck by its jid
                    for t in self.agent.environment.trucks:
                        if(t.jid == truck_jid):
                            cost, path = t.get_shortest_path(self.agent.position)
                            is_busy = t.is_busy                   
                            # update the proposal if meanwhile the truck moved in exploration
                            if(is_busy==False and (cost!=response['cost'] or path!=ast.literal_eval(response['path']))):
                                response['cost']=cost
                                response['path']=f"{path}"
                    
                    if (not is_busy  and
                        (best_proposal is None or                        
                        response['available_capacity'] > best_proposal['available_capacity'] or
                        (response['available_capacity'] == best_proposal['available_capacity'] and response['cost'] < best_proposal['cost']) or
                        (response['available_capacity'] == best_proposal['available_capacity'] and response['cost'] == best_proposal['cost'] and response['fuel'] > best_proposal['fuel']) or
                        (response['available_capacity'] == best_proposal['available_capacity'] and response['cost'] == best_proposal['cost'] and response['fuel'] > best_proposal['fuel']) and response['truck_jid'] < best_proposal['fuel'])):
                        # Update best proposal based on capacity and cost
                        best_proposal = {
                            'truck_jid': truck_jid,
                            'cost': response['cost'],
                            'available_capacity': response['available_capacity'],
                            'fuel': response['fuel'],
                            'path': response['path']
                        }

            if best_proposal:
                # If there's a valid proposal, send acceptance
                if self.agent.current_waste >= 70:
                    await self.accept_best_proposal(best_proposal)
                else:
                    self.agent.is_waiting_for_truck = False
                    self.agent.sent_collection_request = False
            else:
                print(f"[{self.agent.name}] No valid proposal received.")
                self.agent.sent_collection_request = False

        async def accept_best_proposal(self, best_proposal):
            # Send acceptance to truck, including the path in message body
            accept_msg = Message(to = str(best_proposal['truck_jid']))
            accept_msg.set_metadata("performative", "accept")
            path_str = best_proposal['path']
            accept_msg.body = path_str
            await self.send(accept_msg)
            self.agent.is_waiting_for_truck = True
            self.agent.sent_collection_request = False
            print(f"[{self.agent.name}] Acceptance sent to truck {best_proposal['truck_jid']} with path: {best_proposal['path']}.")

    class ReceiveProblemBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout = 1)
            if msg and msg.metadata.get("performative") == "problem":
                # Add a waiting behavior to collect responses from trucks
                wait_for_problem_responses = self.agent.WaitForProblemResolveBehaviour()
                self.agent.add_behaviour(wait_for_problem_responses)
                print("The bin detected a problem")

    class ReceiveProblemResolveBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg:
                if msg.metadata.get("performative") == "resolve-problem":
                    self.agent.resolving += 1

    class WaitForProblemResolveBehaviour(OneShotBehaviour):
        async def run(self):
            # Wait to see if someone comes to resolve the problem
            await asyncio.sleep(3)
            print("Wait time for resolution ended")
            if self.agent.resolving == 0:
                self.agent.is_waiting_for_truck = False
                self.agent.sent_collection_request = False
            self.agent.resolving = 0
            self.kill()

    class GetBinsTimeBehaviour(CyclicBehaviour):
        async def run(self):
            if self.agent.waste == None and self.agent.current_waste >= 0.4*self.agent.max_capacity:
                self.agent.waste = self.agent.current_waste
                self.agent.time = self.agent.environment.timer()
            elif self.agent.waste != None:
                if self.agent.waste > self.agent.current_waste:
                    self.agent.collection_time.append(self.agent.environment.timer() - self.agent.time)
                    self.agent.waste = None
                else:
                    self.agent.waste = self.agent.current_waste


    async def setup(self):
        self.add_behaviour(self.WasteAccumulationBehaviour(period=self.accumulation_period))
        self.add_behaviour(self.ReceiveProposalBehaviour())
        self.add_behaviour(self.ReceiveProblemBehaviour())
        self.add_behaviour(self.ReceiveProblemResolveBehaviour())
        self.add_behaviour(self.GetBinsTimeBehaviour())

        print(f"[{self.name}] Initialized with current waste: {self.current_waste} units.")






# ALERT LEVELS - More priority for fuller bins
# WASTE PRIORITY - busier zones