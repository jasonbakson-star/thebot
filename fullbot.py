import discord
from discord.ext import commands, tasks
import json
import os
import random
import asyncio
import requests
import hashlib
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ====== LOAD ENV ======
load_dotenv()

# ====== CONFIG ======
CONFIG = {
    "BOT_TOKEN": os.getenv("BOT_TOKEN"),
    "NOWPAYMENTS_API_KEY": os.getenv("NOWPAYMENTS_API_KEY"),
    "OWNER_ROLE_ID": int(os.getenv("OWNER_ROLE_ID", 0)),
    "FORTUNE_CASH_PER_DOLLAR": 100,
    "MIN_DEPOSIT": 1.0,
    "HOUSE_TAX": 2,
    "MAX_BET": 1000000,
    "COOLDOWN": 5000,
    "AUTO_WITHDRAW_THRESHOLD": 10000,  # Auto withdraw when user balance hits this
    "AUTO_WITHDRAW_AMOUNT": 5000  # Amount to auto withdraw
}

# ====== DATA MANAGEMENT ======
DATA_FILE = "wine_gambling_data.json"

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        default_data = {
            "users": {},
            "stats": {"total_users": 0, "total_games": 0},
            "cooldowns": {},
            "frozen_users": [],
            "seeds": {
                "current_seed": None,
                "current_seed_id": None,
                "seed_start_time": None,
                "seed_hash": None,
                "revealed_seeds": []
            },
            "pending_withdrawals": {}
        }
        with open(DATA_FILE, "w") as f:
            json.dump(default_data, f, indent=2)
        return default_data

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ====== PROVABLY FAIR SYSTEM ======
def generate_new_seed():
    """Generate a new random seed for provably fair gaming"""
    seed = hashlib.sha256(str(time.time()).encode() + str(random.randint(1, 1000000)).encode()).hexdigest()
    seed_id = random.randint(100, 999)
    seed_hash = hashlib.sha256(seed.encode()).hexdigest()
    return seed, seed_id, seed_hash

def get_current_seed():
    """Get or create current seed"""
    data = load_data()
    current_time = datetime.utcnow()
    
    # Check if we need a new seed (daily rotation)
    if (data["seeds"]["seed_start_time"] is None or 
        datetime.fromisoformat(data["seeds"]["seed_start_time"]) + timedelta(days=1) < current_time):
        
        # Reveal old seed if exists
        if data["seeds"]["current_seed"] is not None:
            data["seeds"]["revealed_seeds"].append({
                "seed_id": data["seeds"]["current_seed_id"],
                "seed": data["seeds"]["current_seed"],
                "hash": data["seeds"]["seed_hash"],
                "start_time": data["seeds"]["seed_start_time"],
                "end_time": current_time.isoformat()
            })
        
        # Generate new seed
        new_seed, seed_id, seed_hash = generate_new_seed()
        data["seeds"]["current_seed"] = new_seed
        data["seeds"]["current_seed_id"] = seed_id
        data["seeds"]["seed_start_time"] = current_time.isoformat()
        data["seeds"]["seed_hash"] = seed_hash
        save_data(data)
    
    return data["seeds"]["current_seed"], data["seeds"]["current_seed_id"], data["seeds"]["seed_hash"]

def provably_fair_random(seed, user_id, nonce, max_value):
    """Generate provably fair random number"""
    combined = f"{seed}-{user_id}-{nonce}"
    hash_result = hashlib.sha256(combined.encode()).hexdigest()
    return int(hash_result[:8], 16) % max_value

# ====== HELPER FUNCTIONS ======
def is_owner(member):
    """Check if user has owner role"""
    return CONFIG.get("OWNER_ROLE_ID", 0) in [role.id for role in member.roles]

def is_user_frozen(user_id):
    """Check if a user is frozen"""
    data = load_data()
    return str(user_id) in data.get("frozen_users", [])

def format_fortune_cash(points):
    return f"{points:,.0f} FC"

# ====== NOWPAYMENTS.IO PAYMENT SYSTEM ======
async def create_payment_link(amount_usd, user_id):
    """Create a payment link using NOWPayments.io API"""
    try:
        order_id = f"DEPO_{user_id}_{int(time.time())}"
        
        # Create invoice (payment link) with NOWPayments
        payment_data = {
            "price_amount": amount_usd,
            "price_currency": "usd",
            "order_id": order_id,
            "order_description": f"Fortune Cash Deposit - {amount_usd * CONFIG['FORTUNE_CASH_PER_DOLLAR']} FC",
            "success_url": "https://discord.com/channels/@me",
            "cancel_url": "https://discord.com/channels/@me"
        }

        headers = {
            'x-api-key': CONFIG["NOWPAYMENTS_API_KEY"],
            'Content-Type': 'application/json'
        }

        response = requests.post(
            'https://api.nowpayments.io/v1/invoice',
            headers=headers,
            json=payment_data
        )

        if response.status_code == 200:
            result = response.json()
            payment_url = result.get('invoice_url')
            invoice_id = result.get('id')
            
            if payment_url:
                # Store pending payment
                data = load_data()
                data["pending_withdrawals"][order_id] = {
                    "user_id": user_id,
                    "amount_usd": amount_usd,
                    "fortune_cash": amount_usd * CONFIG["FORTUNE_CASH_PER_DOLLAR"],
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "pending",
                    "invoice_id": invoice_id
                }
                save_data(data)
                
                return payment_url, order_id
        else:
            print(f"NOWPayments API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Payment creation failed: {e}")
    
    return None, None

# ====== INSTANT WITHDRAW SYSTEM ======
# No manual approval needed - withdrawals are processed instantly

# ====== BOT SETUP ======
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix=".", intents=intents, help_command=None)

# ====== GAME STATE ======
game_state = {
    "coinflip": {},
    "blackjack": {},
    "roulette": {},
    "slots": {},
    "dice": {},
    "rps": {},
    "poker": {},
    "baccarat": {},
    "hilo": {},
    "fight": {}
}

# ====== BUTTON VIEWS ======
class CoinflipView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.used = False

    @discord.ui.button(label="Heads", style=discord.ButtonStyle.primary, emoji="🪙")
    async def heads_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!\n\n*Only the player who started this game can interact with it*",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_coinflip(interaction, "h")

    @discord.ui.button(label="Tails", style=discord.ButtonStyle.secondary, emoji="🪙")
    async def tails_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!\n\n*Only the player who started this game can interact with it*",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_coinflip(interaction, "t")

    async def play_coinflip(self, interaction, choice):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        # Use provably fair random
        fair_result = provably_fair_random(seed, self.user_id, nonce, 2)
        result = "h" if fair_result == 0 else "t"
        
        win = result == choice.lower()
        payout = self.bet * 1.98 if win else 0

        if win:
            user["fortune_cash"] += payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
            user["games_won"] = user.get("games_won", 0) + 1
        
        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        # Removed auto-withdraw - users now withdraw manually

        result_text = "Winnings" if win else "Lost"
        result_amount = payout if win else self.bet
        embed = discord.Embed(
            title="🎉 You Won!" if win else "😢 You Lost!",
            description=f"**Bet:** {format_fortune_cash(self.bet)}\n**Choice:** {choice.upper()}\n**Result:** {result.upper()}\n**{result_text}:** {format_fortune_cash(result_amount)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if win else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

# ====== POKER GAME VIEW ======
class PokerView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=120)
        self.bet = bet
        self.user_id = user_id
        self.used = False
        self.stage = "initial"  # initial, draw, final

    @discord.ui.button(label="Deal Cards", style=discord.ButtonStyle.primary, emoji="🃏")
    async def deal_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if self.stage == "initial":
            await self.deal_initial_cards(interaction)
        
    @discord.ui.button(label="Hold All", style=discord.ButtonStyle.success, emoji="✋")
    async def hold_all_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return
        if self.stage == "draw":
            await self.final_evaluation(interaction, [])

    @discord.ui.button(label="Draw Cards", style=discord.ButtonStyle.secondary, emoji="🔄")
    async def draw_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return
        if self.stage == "draw":
            # For simplicity, draw 3 random cards
            await self.final_evaluation(interaction, [0, 1, 2])

    async def deal_initial_cards(self, interaction):
        # Deal 5 cards for poker
        suits = ["♠", "♣", "♥", "♦"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        
        self.hand = []
        for i in range(5):
            card_idx = provably_fair_random(seed, self.user_id, nonce + i, 52)
            suit_idx = card_idx // 13
            rank_idx = card_idx % 13
            self.hand.append({"rank": ranks[rank_idx], "suit": suits[suit_idx]})
        
        user["game_nonce"] = nonce + 4
        data["users"][str(self.user_id)] = user
        save_data(data)
        
        hand_display = ' '.join(f"{card['rank']}{card['suit']}" for card in self.hand)
        
        self.stage = "draw"
        embed = discord.Embed(
            title="🃏 Poker - Draw Phase",
            description=f"**Your Hand:** {hand_display}\n**Bet:** {format_fortune_cash(self.bet)}\n\nChoose to hold all cards or draw new ones!",
            color=0x0000FF
        )
        await interaction.response.edit_message(embed=embed, view=self)

    async def final_evaluation(self, interaction, cards_to_replace):
        if cards_to_replace:
            # Replace selected cards
            suits = ["♠", "♣", "♥", "♦"]
            ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
            
            data = load_data()
            user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
            seed, seed_id, _ = get_current_seed()
            nonce = user.get("game_nonce", 0) + 1
            
            for i, card_pos in enumerate(cards_to_replace):
                if card_pos < len(self.hand):
                    card_idx = provably_fair_random(seed, self.user_id, nonce + i, 52)
                    suit_idx = card_idx // 13
                    rank_idx = card_idx % 13
                    self.hand[card_pos] = {"rank": ranks[rank_idx], "suit": suits[suit_idx]}
            
            user["game_nonce"] = nonce + len(cards_to_replace) - 1
            data["users"][str(self.user_id)] = user
            save_data(data)

        # Evaluate hand
        hand_strength = self.evaluate_poker_hand()
        payout = self.calculate_poker_payout(hand_strength)
        
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        if payout > 0:
            user["fortune_cash"] += payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        hand_display = ' '.join(f"{card['rank']}{card['suit']}" for card in self.hand)
        
        self.used = True
        embed = discord.Embed(
            title="🎉 Poker Result!" if payout > 0 else "😢 No Winning Hand",
            description=f"**Final Hand:** {hand_display}\n**Hand:** {hand_strength}\n**Payout:** {format_fortune_cash(payout if payout > 0 else 0)}",
            color=0x00FF00 if payout > 0 else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

    def evaluate_poker_hand(self):
        # Simple poker hand evaluation
        ranks = [card["rank"] for card in self.hand]
        suits = [card["suit"] for card in self.hand]
        
        # Check for pairs, three of a kind, etc.
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        if counts == [4, 1]:
            return "Four of a Kind"
        elif counts == [3, 2]:
            return "Full House"
        elif is_flush:
            return "Flush"
        elif counts == [3, 1, 1]:
            return "Three of a Kind"
        elif counts == [2, 2, 1]:
            return "Two Pair"
        elif counts == [2, 1, 1, 1]:
            return "One Pair"
        else:
            return "High Card"

    def calculate_poker_payout(self, hand_strength):
        payouts = {
            "Four of a Kind": self.bet * 25,
            "Full House": self.bet * 9,
            "Flush": self.bet * 6,
            "Three of a Kind": self.bet * 3,
            "Two Pair": self.bet * 2,
            "One Pair": self.bet * 1,
            "High Card": 0
        }
        return payouts.get(hand_strength, 0)

# ====== BACCARAT GAME VIEW ======
class BaccaratView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.used = False

    @discord.ui.button(label="Player", style=discord.ButtonStyle.primary, emoji="👤")
    async def player_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_baccarat(interaction, "player")

    @discord.ui.button(label="Banker", style=discord.ButtonStyle.secondary, emoji="🏦")
    async def banker_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_baccarat(interaction, "banker")

    @discord.ui.button(label="Tie", style=discord.ButtonStyle.success, emoji="🤝")
    async def tie_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_baccarat(interaction, "tie")

    async def play_baccarat(self, interaction, choice):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        # Baccarat card values: A=1, 2-9=face value, 10/J/Q/K=0
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0]  # 13 card values
        
        # Deal two cards each
        p1 = provably_fair_random(seed, self.user_id, nonce, 13)
        p2 = provably_fair_random(seed, self.user_id, nonce + 1, 13)
        b1 = provably_fair_random(seed, self.user_id, nonce + 2, 13)
        b2 = provably_fair_random(seed, self.user_id, nonce + 3, 13)
        
        player_score = (values[p1] + values[p2]) % 10
        banker_score = (values[b1] + values[b2]) % 10
        
        user["game_nonce"] = nonce + 3
        
        # Determine result
        if player_score > banker_score:
            result = "player"
        elif banker_score > player_score:
            result = "banker"
        else:
            result = "tie"

        payout = 0
        if choice == result:
            if result == "player":
                payout = self.bet * 1.98
            elif result == "banker":
                payout = self.bet * 1.95
            else:  # tie
                payout = self.bet * 8

        if payout > 0:
            user["fortune_cash"] += payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        embed = discord.Embed(
            title="🎉 You Won!" if payout > 0 else "😢 You Lost!",
            description=f"**Your Bet:** {choice.title()}\n**Player Score:** {player_score}\n**Banker Score:** {banker_score}\n**Result:** {result.title()}\n**Amount:** {format_fortune_cash(payout if payout > 0 else self.bet)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if payout > 0 else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

# ====== FIGHT GAME VIEW ======
class FightView(discord.ui.View):
    def __init__(self, bet, challenger_id, opponent_id):
        super().__init__(timeout=120)
        self.bet = bet
        self.challenger_id = challenger_id
        self.opponent_id = opponent_id
        self.accepted = False

    @discord.ui.button(label="Accept Fight", style=discord.ButtonStyle.danger, emoji="⚔️")
    async def accept_fight_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.opponent_id:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This challenge is not for you!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if self.accepted:
            return
            
        self.accepted = True
        await self.start_fight(interaction)

    @discord.ui.button(label="Decline", style=discord.ButtonStyle.secondary, emoji="❌")
    async def decline_fight_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.opponent_id:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This challenge is not for you!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        embed = discord.Embed(
            title="🚫 Fight Declined",
            description="The challenge has been declined.",
            color=0xFF6B6B
        )
        await interaction.response.edit_message(embed=embed, view=None)

    async def start_fight(self, interaction):
        data = load_data()
        challenger = data["users"].get(str(self.challenger_id), {"fortune_cash": 0})
        opponent = data["users"].get(str(self.opponent_id), {"fortune_cash": 0})
        
        # Check if opponent has enough funds
        if opponent["fortune_cash"] < self.bet:
            embed = discord.Embed(
                title="💸 Insufficient Funds",
                description="Opponent doesn't have enough Fortune Cash for this fight!",
                color=0xFF6B6B
            )
            await interaction.response.edit_message(embed=embed, view=None)
            return

        # Deduct bet from opponent
        opponent["fortune_cash"] -= self.bet
        
        # Fight simulation using provably fair
        seed, seed_id, _ = get_current_seed()
        nonce = challenger.get("game_nonce", 0) + 1
        
        fight_result = provably_fair_random(seed, self.challenger_id, nonce, 100)
        
        # 60% chance challenger wins, 40% opponent wins (slight advantage to challenger)
        challenger_wins = fight_result < 60
        
        total_pot = self.bet * 2
        winner_payout = total_pot * 0.98  # 2% house edge
        
        if challenger_wins:
            challenger["fortune_cash"] += winner_payout
            challenger["lifetime_earnings"] = challenger.get("lifetime_earnings", 0) + winner_payout
            challenger["games_won"] = challenger.get("games_won", 0) + 1
            winner_name = f"<@{self.challenger_id}>"
        else:
            opponent["fortune_cash"] += winner_payout
            opponent["lifetime_earnings"] = opponent.get("lifetime_earnings", 0) + winner_payout
            opponent["games_won"] = opponent.get("games_won", 0) + 1
            winner_name = f"<@{self.opponent_id}>"
        
        challenger["game_nonce"] = nonce
        challenger["lifetime_wagered"] = challenger.get("lifetime_wagered", 0) + self.bet
        challenger["games_played"] = challenger.get("games_played", 0) + 1
        
        opponent["lifetime_wagered"] = opponent.get("lifetime_wagered", 0) + self.bet
        opponent["games_played"] = opponent.get("games_played", 0) + 1
        
        data["stats"]["total_games"] += 1
        data["users"][str(self.challenger_id)] = challenger
        data["users"][str(self.opponent_id)] = opponent
        save_data(data)

        embed = discord.Embed(
            title="⚔️ Fight Results!",
            description=f"**Winner:** {winner_name}\n**Prize:** {format_fortune_cash(winner_payout)}\n**Fight Value:** {fight_result}/100\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00
        )
        await interaction.response.edit_message(embed=embed, view=None)

# ====== ADDITIONAL GAME VIEWS ======
class RouletteView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.used = False

    @discord.ui.button(label="Red", style=discord.ButtonStyle.danger, emoji="🔴")
    async def red_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_roulette(interaction, "red")

    @discord.ui.button(label="Black", style=discord.ButtonStyle.secondary, emoji="⚫")
    async def black_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_roulette(interaction, "black")

    @discord.ui.button(label="Green (0)", style=discord.ButtonStyle.success, emoji="🟢")
    async def green_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_roulette(interaction, "0")

    async def play_roulette(self, interaction, choice):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        # Use provably fair random
        result = provably_fair_random(seed, self.user_id, nonce, 37)  # 0-36
        color = "green" if result == 0 else "red" if result % 2 == 1 else "black"
        
        win = False
        payout = 0

        if choice.lower() == "red" and color == "red":
            win = True
            payout = self.bet * 2
        elif choice.lower() == "black" and color == "black":
            win = True
            payout = self.bet * 2
        elif choice == "0" and result == 0:
            win = True
            payout = self.bet * 36

        if win:
            taxed_payout = payout * 0.98
            user["fortune_cash"] += taxed_payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + taxed_payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        color_emoji = "🔴" if color == "red" else "⚫" if color == "black" else "🟢"
        result_text = "Winnings" if win else "Lost"
        result_amount = taxed_payout if win else self.bet
        embed = discord.Embed(
            title="🎉 You Won!" if win else "😢 You Lost!",
            description=f"**Bet:** {choice} ({format_fortune_cash(self.bet)})\n**Result:** {result} {color_emoji} ({color})\n**{result_text}:** {format_fortune_cash(result_amount)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if win else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

class SlotsView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.used = False

    @discord.ui.button(label="🎰 SPIN 🎰", style=discord.ButtonStyle.primary, emoji="🎰")
    async def spin_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            embed = discord.Embed(
                title="🚫 Access Denied",
                description="This is not your game!",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        self.used = True
        await self.play_slots(interaction)

    async def play_slots(self, interaction):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        symbols = ["🍒", "🍋", "🍊", "🍇", "🍎", "⭐", "💎", "🔔"]
        
        # Generate 3 reels using provably fair random
        reel1 = symbols[provably_fair_random(seed, self.user_id, nonce, len(symbols))]
        reel2 = symbols[provably_fair_random(seed, self.user_id, nonce + 1, len(symbols))]
        reel3 = symbols[provably_fair_random(seed, self.user_id, nonce + 2, len(symbols))]
        
        user["game_nonce"] = nonce + 2
        
        # Calculate winnings
        payout = 0
        if reel1 == reel2 == reel3:
            if reel1 == "💎":
                payout = self.bet * 30
            elif reel1 == "⭐":
                payout = self.bet * 20
            elif reel1 == "🔔":
                payout = self.bet * 15
            else:
                payout = self.bet * 10
        elif reel1 == reel2 or reel2 == reel3 or reel1 == reel3:
            payout = self.bet * 2

        if payout > 0:
            taxed_payout = payout * 0.98
            user["fortune_cash"] += taxed_payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + taxed_payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        result_text = "Winnings" if payout > 0 else "Lost"
        result_amount = taxed_payout if payout > 0 else self.bet
        embed = discord.Embed(
            title="🎉 You Won!" if payout > 0 else "😢 You Lost!",
            description=f"**Reels:** {reel1} {reel2} {reel3}\n**{result_text}:** {format_fortune_cash(result_amount)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if payout > 0 else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

class RPSView(discord.ui.View):
    def __init__(self, bet, user_id):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.used = False

    @discord.ui.button(label="Rock", style=discord.ButtonStyle.secondary, emoji="🪨")
    async def rock_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_rps(interaction, "rock")

    @discord.ui.button(label="Paper", style=discord.ButtonStyle.primary, emoji="📄")
    async def paper_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_rps(interaction, "paper")

    @discord.ui.button(label="Scissors", style=discord.ButtonStyle.danger, emoji="✂️")
    async def scissors_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_rps(interaction, "scissors")

    async def play_rps(self, interaction, choice):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        choices = ["rock", "paper", "scissors"]
        bot_choice = choices[provably_fair_random(seed, self.user_id, nonce, 3)]
        
        # Determine winner
        if choice == bot_choice:
            result = "tie"
            payout = self.bet  # Return bet
        elif (choice == "rock" and bot_choice == "scissors") or \
             (choice == "paper" and bot_choice == "rock") or \
             (choice == "scissors" and bot_choice == "paper"):
            result = "win"
            payout = self.bet * 1.98
        else:
            result = "lose"
            payout = 0

        if payout > 0:
            user["fortune_cash"] += payout
            if result == "win":
                user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
                user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        emojis = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
        
        embed = discord.Embed(
            title=f"🎉 You Won!" if result == "win" else "🤝 Tie!" if result == "tie" else "😢 You Lost!",
            description=f"**You:** {emojis[choice]} {choice.title()}\n**Bot:** {emojis[bot_choice]} {bot_choice.title()}\n**Result:** {format_fortune_cash(payout if payout > 0 else self.bet)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if result == "win" else 0xFFFF00 if result == "tie" else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

class HiLoView(discord.ui.View):
    def __init__(self, bet, user_id, current_card):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.current_card = current_card
        self.used = False

    @discord.ui.button(label="Higher", style=discord.ButtonStyle.success, emoji="⬆️")
    async def higher_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_hilo(interaction, "higher")

    @discord.ui.button(label="Lower", style=discord.ButtonStyle.danger, emoji="⬇️")
    async def lower_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_hilo(interaction, "lower")

    async def play_hilo(self, interaction, guess):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        # Generate next card
        suits = ["♠", "♣", "♥", "♦"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        
        card_idx = provably_fair_random(seed, self.user_id, nonce, 52)
        suit_idx = card_idx // 13
        rank_idx = card_idx % 13
        
        next_card = {
            "rank": ranks[rank_idx],
            "suit": suits[suit_idx],
            "value": values[rank_idx]
        }
        
        # Determine winner
        win = False
        if guess == "higher" and next_card["value"] > self.current_card["value"]:
            win = True
        elif guess == "lower" and next_card["value"] < self.current_card["value"]:
            win = True
        elif next_card["value"] == self.current_card["value"]:
            win = True  # Tie = win for player

        payout = self.bet * 1.98 if win else 0

        if win:
            user["fortune_cash"] += payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        embed = discord.Embed(
            title="🎉 You Won!" if win else "😢 You Lost!",
            description=f"**Previous:** {self.current_card['rank']}{self.current_card['suit']} ({self.current_card['value']})\n**Next:** {next_card['rank']}{next_card['suit']} ({next_card['value']})\n**Guess:** {guess.title()}\n**Result:** {format_fortune_cash(payout if win else self.bet)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if win else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

class DiceView(discord.ui.View):
    def __init__(self, bet, user_id, sides, prediction, target):
        super().__init__(timeout=60)
        self.bet = bet
        self.user_id = user_id
        self.sides = sides
        self.prediction = prediction
        self.target = target
        self.used = False

    @discord.ui.button(label="🎲 ROLL DICE", style=discord.ButtonStyle.primary, emoji="🎲")
    async def roll_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id or self.used:
            return
        self.used = True
        await self.play_dice(interaction)

    async def play_dice(self, interaction):
        data = load_data()
        user = data["users"].get(str(self.user_id), {"fortune_cash": 0})
        
        seed, seed_id, _ = get_current_seed()
        nonce = user.get("game_nonce", 0) + 1
        user["game_nonce"] = nonce
        
        # Roll dice using provably fair random
        result = provably_fair_random(seed, self.user_id, nonce, self.sides) + 1
        
        win = False
        if self.prediction == "over" and result > self.target:
            win = True
        elif self.prediction == "under" and result < self.target:
            win = True
        elif self.prediction == "exact" and result == self.target:
            win = True

        payout = 0
        if win:
            if self.prediction == "exact":
                payout = self.bet * self.sides
            else:
                payout = self.bet * 1.98

        if win:
            user["fortune_cash"] += payout
            user["lifetime_earnings"] = user.get("lifetime_earnings", 0) + payout
            user["games_won"] = user.get("games_won", 0) + 1

        data["stats"]["total_games"] += 1
        data["users"][str(self.user_id)] = user
        save_data(data)

        embed = discord.Embed(
            title="🎉 You Won!" if win else "😢 You Lost!",
            description=f"**Dice:** {self.sides}-sided\n**Roll:** {result}\n**Prediction:** {self.prediction} {self.target}\n**Result:** {'WIN' if win else 'LOSE'}\n**Amount:** {format_fortune_cash(payout if win else self.bet)}\n\n**Seed ID:** {seed_id} | **Nonce:** {nonce}",
            color=0x00FF00 if win else 0xFF0000
        )
        await interaction.response.edit_message(embed=embed, view=None)

# ====== BOT EVENTS ======
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')
    await bot.change_presence(activity=discord.Game(name="🍷 .help | Gambling Bot"))

# ====== BOT COMMANDS ======
@bot.command(name="depo")
async def handle_deposit(ctx, amount: float = None):
    """Create payment link for deposit"""
    if amount is None:
        embed = discord.Embed(
            title="❌ Missing Amount",
            description="**Usage:** `.depo <amount>`\n**Example:** `.depo 10` (for $10 USD)\n\n*Please specify how much you want to deposit*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
        
    if amount < CONFIG["MIN_DEPOSIT"]:
        embed = discord.Embed(
            title="❌ Invalid Amount",
            description=f"**Minimum deposit:** ${CONFIG['MIN_DEPOSIT']}\n**Your amount:** ${amount}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    payment_url, order_id = await create_payment_link(amount, ctx.author.id)
    
    if payment_url:
        fortune_cash = amount * CONFIG["FORTUNE_CASH_PER_DOLLAR"]
        embed = discord.Embed(
            title="💳 Payment Link Created",
            description=f"**Amount:** ${amount} USD\n**Fortune Cash:** {format_fortune_cash(fortune_cash)}\n**Order ID:** {order_id}\n\n[**Click here to pay**]({payment_url})\n\n*Payment link expires in 30 minutes*",
            color=0x00FF00
        )
        await ctx.reply(embed=embed)
    else:
        embed = discord.Embed(
            title="❌ Payment Error",
            description="Failed to create payment link. Please try again later.",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)

@bot.command(name="bal", aliases=["balance"])
async def handle_balance(ctx):
    """Check user balance and stats"""
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})
    
    _, seed_id, seed_hash = get_current_seed()
    
    embed = discord.Embed(
        title="💰 Your Balance",
        description=f"**Fortune Cash:** {format_fortune_cash(user['fortune_cash'])}\n**USD Value:** ${user['fortune_cash'] / CONFIG['FORTUNE_CASH_PER_DOLLAR']:.2f}",
        color=0x722F37
    )
    embed.add_field(
        name="📊 Stats",
        value=f"**Games Played:** {user.get('games_played', 0)}\n**Games Won:** {user.get('games_won', 0)}\n**Lifetime Wagered:** {format_fortune_cash(user.get('lifetime_wagered', 0))}\n**Lifetime Earnings:** {format_fortune_cash(user.get('lifetime_earnings', 0))}",
        inline=False
    )
    embed.add_field(
        name="🎲 Provably Fair",
        value=f"**Current Seed ID:** {seed_id}\n**Seed Hash:** {seed_hash[:16]}...\n**Your Nonce:** {user.get('game_nonce', 0)}",
        inline=False
    )
    await ctx.reply(embed=embed)

@bot.command()
async def help(ctx):
    embed = discord.Embed(
        title="🎰 Fortune Bot",
        description="**100 FC = $1 USD | Provably Fair Gaming**",
        color=0x722F37
    )
    embed.add_field(
        name="💰 Economy", 
        value="`.depo <amount>` - Create payment link\n`.bal` - Check balance & seed info\n`.withdraw <amount> <address>` - Withdraw to crypto address", 
        inline=False
    )
    embed.add_field(
        name="🎮 Button Games", 
        value="`.cf <amount>` - Coinflip (buttons)\n`.bc <amount>` - Baccarat (buttons)\n`.pk <amount>` - Poker (buttons)\n`.rl <amount>` - Roulette (buttons)\n`.sl <amount>` - Slots (buttons)\n`.rps <amount>` - Rock Paper Scissors\n`.hl <amount>` - Hi-Lo (buttons)", 
        inline=False
    )
    embed.add_field(
        name="🎯 Advanced Games",
        value="`.di <amount> <sides> <over/under/exact> <target>` - Dice\n`.ft <amount> @user` - Fight another user",
        inline=False
    )
    embed.add_field(
        name="🎲 Provably Fair", 
        value="`.seed` - View current & revealed seeds\n`.verify <seed_id> <nonce> <result>` - Verify results", 
        inline=False
    )
    if is_owner(ctx.author):
        embed.add_field(
            name="👑 Admin Access", 
            value="Use `.cmds` to view admin commands", 
            inline=False
        )
    embed.set_footer(text="Enhanced Bot | 2% House Edge | Provably Fair | Instant Withdrawals")
    await ctx.reply(embed=embed)

@bot.command(name="cf", aliases=["coinflip"])
async def handle_coinflip(ctx, bet: float):
    """Coinflip game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.cf <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🪙 Coinflip",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Choose:** Heads or Tails",
        color=0x0000FF
    )
    
    view = CoinflipView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="bc", aliases=["baccarat"])
async def handle_baccarat(ctx, bet: float):
    """Baccarat game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.bc <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🎴 Baccarat",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Choose:** Player (1.98x), Banker (1.95x), or Tie (8x)",
        color=0x0000FF
    )
    
    view = BaccaratView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="pk", aliases=["poker"])
async def handle_poker(ctx, bet: float):
    """5-card draw poker game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.pk <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🃏 5-Card Draw Poker",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Payouts:** Four of a Kind (25x), Full House (9x), Flush (6x), Three of a Kind (3x), Two Pair (2x), One Pair (1x)",
        color=0x0000FF
    )
    
    view = PokerView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="ft", aliases=["fight"])
async def handle_fight(ctx, bet: float, opponent: discord.Member):
    """Challenge another user to a fight"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return

    if opponent.id == ctx.author.id:
        embed = discord.Embed(
            title="❌ Invalid Target",
            description="You cannot fight yourself!",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.ft <amount> @user`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    # Deduct bet from challenger
    user["fortune_cash"] -= bet
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="⚔️ Fight Challenge!",
        description=f"**Challenger:** {ctx.author.mention}\n**Opponent:** {opponent.mention}\n**Bet:** {format_fortune_cash(bet)}\n\n{opponent.mention}, do you accept this challenge?",
        color=0xFF6B6B
    )
    
    view = FightView(bet, ctx.author.id, opponent.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="seed")
async def handle_seed(ctx):
    """View current and revealed seeds"""
    data = load_data()
    _, seed_id, seed_hash = get_current_seed()
    
    embed = discord.Embed(
        title="🎲 Provably Fair Seeds",
        description=f"**Current Seed ID:** {seed_id}\n**Seed Hash:** {seed_hash}",
        color=0x722F37
    )
    
    # Show last 3 revealed seeds
    revealed = data["seeds"]["revealed_seeds"][-3:]
    if revealed:
        seed_text = ""
        for seed_info in revealed:
            seed_text += f"**ID {seed_info['seed_id']}:** {seed_info['seed'][:32]}...\n"
        embed.add_field(name="🔓 Recently Revealed Seeds", value=seed_text, inline=False)
    
    await ctx.reply(embed=embed)

@bot.command(name="rl", aliases=["roulette"])
async def handle_roulette(ctx, bet: float):
    """Roulette game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.rl <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🎯 Roulette",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Choose:** Red (2x), Black (2x), or Green 0 (36x)",
        color=0x0000FF
    )
    
    view = RouletteView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="sl", aliases=["slots"])
async def handle_slots(ctx, bet: float):
    """Slots game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.sl <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🎰 Slots",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Payouts:** 💎💎💎 = 30x | ⭐⭐⭐ = 20x | 🔔🔔🔔 = 15x\nOther triples = 10x | Pairs = 2x",
        color=0x0000FF
    )
    
    view = SlotsView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="rps")
async def handle_rps(ctx, bet: float):
    """Rock Paper Scissors game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.rps <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🪨📄✂️ Rock Paper Scissors",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Choose your weapon!**",
        color=0x0000FF
    )
    
    view = RPSView(bet, ctx.author.id)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="hl", aliases=["hilo"])
async def handle_hilo(ctx, bet: float):
    """Hi-Lo card game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.hl <amount>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    suits = ["♠", "♣", "♥", "♦"]
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    card_idx = random.randint(0, 51)
    suit_idx = card_idx // 13
    rank_idx = card_idx % 13
    
    current_card = {
        "rank": ranks[rank_idx],
        "suit": suits[suit_idx],
        "value": values[rank_idx]
    }

    embed = discord.Embed(
        title="🃏 Hi-Lo",
        description=f"**Current Card:** {current_card['rank']}{current_card['suit']}\n**Bet:** {format_fortune_cash(bet)}\n**Guess:** Will the next card be higher or lower?",
        color=0x0000FF
    )
    
    view = HiLoView(bet, ctx.author.id, current_card)
    await ctx.reply(embed=embed, view=view)

@bot.command(name="di", aliases=["dice"])
async def handle_dice(ctx, bet: float, sides: int, prediction: str, target: int):
    """Dice game"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})

    if bet <= 0 or bet > CONFIG["MAX_BET"]:
        embed = discord.Embed(
            title="❌ Invalid Bet Amount",
            description=f"**Usage:** `.di <amount> <sides> <over/under/exact> <target>`\n**Max Bet:** {format_fortune_cash(CONFIG['MAX_BET'])}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if sides < 2 or sides > 100:
        embed = discord.Embed(
            title="🎲 Invalid Dice Configuration",
            description=f"**Sides Range:** 2-100\n**Your Input:** {sides} sides",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    if prediction.lower() not in ["over", "under", "exact"]:
        embed = discord.Embed(
            title="🎯 Invalid Prediction Type",
            description=f"**Valid Options:** `over`, `under`, `exact`\n**Your Input:** `{prediction}`",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    if target < 1 or target > sides:
        embed = discord.Embed(
            title="🎯 Invalid Target Number",
            description=f"**Valid Range:** 1-{sides}\n**Your Target:** {target}",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    if user["fortune_cash"] < bet:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(bet)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n*Use `.depo <amount>` to add funds*",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return

    user["fortune_cash"] -= bet
    user["lifetime_wagered"] = user.get("lifetime_wagered", 0) + bet
    user["games_played"] = user.get("games_played", 0) + 1
    data["users"][user_id] = user
    save_data(data)

    embed = discord.Embed(
        title="🎲 Dice Game",
        description=f"**Bet:** {format_fortune_cash(bet)}\n**Dice:** {sides}-sided\n**Prediction:** {prediction} {target}\n**Click to roll!**",
        color=0x0000FF
    )
    
    view = DiceView(bet, ctx.author.id, sides, prediction, target)
    await ctx.reply(embed=embed, view=view)

# ====== ADMIN COMMANDS ======
@bot.command(name="add")
async def handle_add_points(ctx, user: discord.Member, amount: float):
    """Add wine points to a user"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if amount <= 0:
        embed = discord.Embed(
            title="❌ Invalid Amount",
            description=f"**Amount:** {format_fortune_cash(amount)}\n\n*Amount must be greater than 0*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    data = load_data()
    user_id = str(user.id)
    if user_id not in data["users"]:
        data["users"][user_id] = {"fortune_cash": 0}
    
    data["users"][user_id]["fortune_cash"] += amount
    save_data(data)

    embed = discord.Embed(
        title="✅ Fortune Cash Added",
        description=f"**User:** {user.mention}\n**Amount Added:** {format_fortune_cash(amount)}\n**New Balance:** {format_fortune_cash(data['users'][user_id]['fortune_cash'])}",
        color=0x00FF00,
        timestamp=datetime.utcnow()
    )
    await ctx.reply(embed=embed)

@bot.command(name="remove")
async def handle_remove_points(ctx, user: discord.Member, amount: float):
    """Remove wine points from a user"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    if amount <= 0:
        embed = discord.Embed(
            title="❌ Invalid Amount",
            description=f"**Amount:** {format_fortune_cash(amount)}\n\n*Amount must be greater than 0*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    data = load_data()
    user_id = str(user.id)
    if user_id not in data["users"]:
        data["users"][user_id] = {"fortune_cash": 0}
    
    if data["users"][user_id]["fortune_cash"] < amount:
        embed = discord.Embed(
            title="⚠️ Insufficient Balance",
            description=f"**User:** {user.mention}\n**Current Balance:** {format_fortune_cash(data['users'][user_id]['fortune_cash'])}\n**Attempted Removal:** {format_fortune_cash(amount)}\n\n*Cannot remove more than user has*",
            color=0xFFA500
        )
        await ctx.reply(embed=embed)
        return
    
    data["users"][user_id]["fortune_cash"] -= amount
    save_data(data)

    embed = discord.Embed(
        title="✅ Fortune Cash Removed",
        description=f"**User:** {user.mention}\n**Amount Removed:** {format_fortune_cash(amount)}\n**New Balance:** {format_fortune_cash(data['users'][user_id]['fortune_cash'])}",
        color=0xFF6B6B,
        timestamp=datetime.utcnow()
    )
    await ctx.reply(embed=embed)

@bot.command(name="freeze")
async def handle_freeze_user(ctx, user: discord.Member):
    """Freeze a user account"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    data = load_data()
    user_id = str(user.id)
    
    if user_id in data.get("frozen_users", []):
        embed = discord.Embed(
            title="❄️ Already Frozen",
            description=f"**User:** {user.mention}\n\n*This user is already frozen*",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
    
    if "frozen_users" not in data:
        data["frozen_users"] = []
    
    data["frozen_users"].append(user_id)
    save_data(data)

    embed = discord.Embed(
        title="🧊 User Frozen",
        description=f"**User:** {user.mention}\n**Status:** Account frozen\n\n*User cannot access gambling commands*",
        color=0x0099FF,
        timestamp=datetime.utcnow()
    )
    await ctx.reply(embed=embed)

@bot.command(name="unfreeze")
async def handle_unfreeze_user(ctx, user: discord.Member):
    """Unfreeze a user account"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return

    data = load_data()
    user_id = str(user.id)
    
    if user_id not in data.get("frozen_users", []):
        embed = discord.Embed(
            title="🔥 Not Frozen",
            description=f"**User:** {user.mention}\n\n*This user is not frozen*",
            color=0xFFA500
        )
        await ctx.reply(embed=embed)
        return
    
    data["frozen_users"].remove(user_id)
    save_data(data)

    embed = discord.Embed(
        title="🔥 User Unfrozen",
        description=f"**User:** {user.mention}\n**Status:** Account unfrozen\n\n*User can now access gambling commands*",
        color=0x00FF8A,
        timestamp=datetime.utcnow()
    )
    await ctx.reply(embed=embed)

@bot.command(name="withdraw")
async def handle_withdraw(ctx, amount: float, address: str):
    """Instant withdrawal command"""
    if is_user_frozen(ctx.author.id):
        embed = discord.Embed(
            title="🧊 Account Frozen",
            description="Your account has been frozen by an administrator.",
            color=0x0099FF
        )
        await ctx.reply(embed=embed)
        return
        
    data = load_data()
    user_id = str(ctx.author.id)
    user = data["users"].get(user_id, {"fortune_cash": 0})
    
    min_withdraw = 100  # Minimum 100 WP ($1) withdrawal
    
    if amount < min_withdraw:
        embed = discord.Embed(
            title="❌ Minimum Withdrawal",
            description=f"**Minimum withdrawal:** {format_fortune_cash(min_withdraw)} ($1.00)\n**Your amount:** {format_fortune_cash(amount)}",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    if user["fortune_cash"] < amount:
        embed = discord.Embed(
            title="💸 Insufficient Funds",
            description=f"**Required:** {format_fortune_cash(amount)}\n**Your Balance:** {format_fortune_cash(user['fortune_cash'])}",
            color=0xFF6B6B
        )
        await ctx.reply(embed=embed)
        return
    
    # Process instant withdrawal
    user["fortune_cash"] -= amount
    usd_amount = amount / CONFIG["FORTUNE_CASH_PER_DOLLAR"]
    
    # Add to user's withdrawal history
    if "withdrawals" not in user:
        user["withdrawals"] = []
    
    user["withdrawals"].append({
        "amount_wp": amount,
        "amount_usd": usd_amount,
        "address": address,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Update global stats
    if "total_withdrawals" not in data["stats"]:
        data["stats"]["total_withdrawals"] = 0
        data["stats"]["total_withdrawn_usd"] = 0
    
    data["stats"]["total_withdrawals"] += 1
    data["stats"]["total_withdrawn_usd"] += usd_amount
    
    data["users"][user_id] = user
    save_data(data)
    
    embed = discord.Embed(
        title="💰 Withdrawal Processed!",
        description=f"**Amount:** {format_fortune_cash(amount)} (${usd_amount:.2f})\n**Address:** `{address}`\n**New Balance:** {format_fortune_cash(user['fortune_cash'])}\n\n✅ **Your withdrawal has been processed instantly!**",
        color=0x00FF00,
        timestamp=datetime.utcnow()
    )
    await ctx.reply(embed=embed)

@bot.command(name="house")
async def handle_house_balance(ctx):
    """Check house balance"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    data = load_data()
    house_balance = 0
    
    # Calculate house balance from all games played
    for user_data in data["users"].values():
        house_balance += user_data.get("lifetime_wagered", 0) * 0.02  # 2% house edge
    
    total_games = data["stats"].get("total_games", 0)
    total_users = len(data["users"])
    
    embed = discord.Embed(
        title="🏠 House Statistics",
        description=f"**House Balance:** {format_fortune_cash(house_balance)} (${house_balance / CONFIG['FORTUNE_CASH_PER_DOLLAR']:.2f})\n**Total Games:** {total_games:,}\n**Total Users:** {total_users:,}",
        color=0x722F37,
        timestamp=datetime.utcnow()
    )
    
    # Show total withdrawals processed
    total_withdrawals = data["stats"].get("total_withdrawals", 0)
    total_withdrawn_usd = data["stats"].get("total_withdrawn_usd", 0)
    
    if total_withdrawals > 0:
        embed.add_field(
            name="💰 Total Withdrawals Processed", 
            value=f"**Count:** {total_withdrawals:,}\n**Total Amount:** ${total_withdrawn_usd:.2f}", 
            inline=False
        )
    
    await ctx.reply(embed=embed)

@bot.command(name="transactions")
async def handle_transactions(ctx, user: discord.Member):
    """View user transaction history (Admin only)"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    data = load_data()
    user_id = str(user.id)
    user_data = data["users"].get(user_id, {})
    
    if not user_data:
        embed = discord.Embed(
            title="❌ User Not Found",
            description=f"**User:** {user.mention}\n\n*No gambling data found for this user*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    embed = discord.Embed(
        title=f"📊 Transaction History - {user.display_name}",
        description=f"**Current Balance:** {format_fortune_cash(user_data.get('fortune_cash', 0))}\n**User ID:** {user.id}",
        color=0x722F37,
        timestamp=datetime.utcnow()
    )
    
    # Game stats
    total_games = user_data.get("total_games", 0)
    total_wagered = user_data.get("lifetime_wagered", 0)
    total_won = user_data.get("lifetime_won", 0)
    net_profit = total_won - total_wagered
    
    embed.add_field(
        name="🎮 Gaming Stats",
        value=f"**Total Games:** {total_games:,}\n**Total Wagered:** {format_fortune_cash(total_wagered)}\n**Total Won:** {format_fortune_cash(total_won)}\n**Net P&L:** {format_fortune_cash(net_profit)}",
        inline=True
    )
    
    # Withdrawal history
    withdrawals = user_data.get("withdrawals", [])
    if withdrawals:
        total_withdrawn = sum(w["amount_wp"] for w in withdrawals)
        recent_withdrawals = withdrawals[-3:]  # Last 3 withdrawals
        
        withdrawal_text = f"**Total Withdrawn:** {format_fortune_cash(total_withdrawn)}\n**Count:** {len(withdrawals)}\n\n**Recent:**\n"
        for w in recent_withdrawals:
            date = datetime.fromisoformat(w["timestamp"]).strftime("%m/%d %H:%M")
            withdrawal_text += f"• {format_fortune_cash(w['amount_wp'])} ({date})\n"
    else:
        withdrawal_text = "No withdrawals yet"
    
    embed.add_field(
        name="💰 Withdrawals",
        value=withdrawal_text,
        inline=True
    )
    
    # Account status
    status = "🧊 Frozen" if is_user_frozen(user.id) else "✅ Active"
    embed.add_field(
        name="📋 Account Status",
        value=f"**Status:** {status}\n**Seed ID:** {user_data.get('current_seed_id', 0)}\n**Games Today:** {user_data.get('games_today', 0)}",
        inline=True
    )
    
    await ctx.reply(embed=embed)

@bot.command(name="cmds")
async def handle_admin_commands(ctx):
    """Show admin commands (Admin only)"""
    if not is_owner(ctx.author):
        embed = discord.Embed(
            title="🚫 Access Denied",
            description="You don't have permission to use this command.\n\n*Owner role required*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    embed = discord.Embed(
        title="👑 Admin Commands",
        description="**Available admin-only commands:**",
        color=0x722F37,
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(
        name="💰 User Management",
        value="`.add <user> <amount>` - Add Fortune Cash to user\n`.remove <user> <amount>` - Remove Fortune Cash from user\n`.transactions <user>` - View user transaction history",
        inline=False
    )
    
    embed.add_field(
        name="🔒 Account Control",
        value="`.freeze <user>` - Freeze user account\n`.unfreeze <user>` - Unfreeze user account",
        inline=False
    )
    
    embed.add_field(
        name="📊 Statistics",
        value="`.house` - View house balance & stats",
        inline=False
    )
    
    embed.set_footer(text="Admin Commands | Owner Role Required")
    await ctx.reply(embed=embed)

@bot.command(name="verify")
async def handle_verify(ctx, seed_id: int, nonce: int, expected_result: int):
    """Verify a provably fair result"""
    data = load_data()
    
    # Find the seed
    target_seed = None
    for revealed in data["seeds"]["revealed_seeds"]:
        if revealed["seed_id"] == seed_id:
            target_seed = revealed["seed"]
            break
    
    if not target_seed:
        embed = discord.Embed(
            title="❌ Seed Not Found",
            description=f"**Seed ID:** {seed_id}\n\n*This seed has not been revealed yet or doesn't exist*",
            color=0xFF0000
        )
        await ctx.reply(embed=embed)
        return
    
    # Verify the result
    actual_result = provably_fair_random(target_seed, ctx.author.id, nonce, 100)
    
    embed = discord.Embed(
        title="🔍 Provably Fair Verification",
        description=f"**Seed ID:** {seed_id}\n**Nonce:** {nonce}\n**Expected Result:** {expected_result}\n**Actual Result:** {actual_result}\n**Status:** {'✅ VERIFIED' if actual_result == expected_result else '❌ MISMATCH'}",
        color=0x00FF00 if actual_result == expected_result else 0xFF0000
    )
    await ctx.reply(embed=embed)

# Run the bot
if __name__ == "__main__":
    bot.run(CONFIG["BOT_TOKEN"])