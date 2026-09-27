"""Command-Line UNO Game Simulator with 4 Players and Strategic AI Agents.

Simulates 1,000 games of UNO between 4 players:
- Players 1, 2, 3: RandomStrategy (plays a random valid card, or draws if none)
- Player 0: ColorMatchPriorityStrategy (prioritizes matching current color to control flow)

At the end of the simulation, prints comprehensive statistics including:
- Win counts and win percentages per player
- Average turns per game
- Total cards drawn and action cards triggered
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


COLORS = ["RED", "BLUE", "GREEN", "YELLOW"]
VALUES = [str(i) for i in range(10)] + ["SKIP", "REVERSE", "DRAW_TWO"]
WILDS = ["WILD", "WILD_DRAW_FOUR"]


@dataclass(frozen=True)
class Card:
    color: str  # "RED", "BLUE", "GREEN", "YELLOW", or "WILD"
    value: str  # "0"-"9", "SKIP", "REVERSE", "DRAW_TWO", "WILD", "WILD_DRAW_FOUR"

    def is_wild(self) -> bool:
        return self.color == "WILD" or self.value in WILDS

    def can_play_on(self, active_color: str, active_value: str) -> bool:
        if self.is_wild():
            return True
        return self.color == active_color or self.value == active_value

    def __str__(self) -> str:
        return f"[{self.color}:{self.value}]"


def build_deck() -> List[Card]:
    deck: List[Card] = []
    for color in COLORS:
        deck.append(Card(color=color, value="0"))
        for val in VALUES:
            if val != "0":
                deck.append(Card(color=color, value=val))
                deck.append(Card(color=color, value=val))
    for _ in range(4):
        deck.append(Card(color="WILD", value="WILD"))
        deck.append(Card(color="WILD", value="WILD_DRAW_FOUR"))
    random.shuffle(deck)
    return deck


class PlayerStrategy:
    """Base interface for UNO player decision-making."""

    def __init__(self, name: str):
        self.name = name

    def choose_card(
        self,
        hand: List[Card],
        active_color: str,
        active_value: str,
    ) -> Optional[Tuple[Card, str]]:
        """Select a card to play and declared active color. Returns None to draw."""
        raise NotImplementedError


class RandomStrategy(PlayerStrategy):
    """Plays a randomly chosen valid card from hand."""

    def __init__(self, name: str = "RandomStrategy"):
        super().__init__(name)

    def choose_card(
        self,
        hand: List[Card],
        active_color: str,
        active_value: str,
    ) -> Optional[Tuple[Card, str]]:
        valid = [c for c in hand if c.can_play_on(active_color, active_value)]
        if not valid:
            return None
        chosen = random.choice(valid)
        if chosen.is_wild():
            # Pick color most common in remaining hand
            color_counts = {c: sum(1 for card in hand if card.color == c) for c in COLORS}
            best_color = max(color_counts, key=color_counts.get) or "RED"
            return chosen, best_color
        return chosen, chosen.color


class ColorMatchPriorityStrategy(PlayerStrategy):
    """Prioritizes cards matching active color, saves action cards and wilds for defense."""

    def __init__(self, name: str = "ColorMatchPriority"):
        super().__init__(name)

    def choose_card(
        self,
        hand: List[Card],
        active_color: str,
        active_value: str,
    ) -> Optional[Tuple[Card, str]]:
        valid = [c for c in hand if c.can_play_on(active_color, active_value)]
        if not valid:
            return None

        # 1. Prefer number cards matching active color
        color_numbers = [c for c in valid if c.color == active_color and c.value.isdigit()]
        if color_numbers:
            return color_numbers[0], active_color

        # 2. Prefer action cards matching active color
        color_actions = [c for c in valid if c.color == active_color and not c.value.isdigit()]
        if color_actions:
            return color_actions[0], active_color

        # 3. Prefer matching value cards in other colors
        value_matches = [c for c in valid if not c.is_wild() and c.value == active_value]
        if value_matches:
            c = value_matches[0]
            return c, c.color

        # 4. Fallback to wilds
        wilds = [c for c in valid if c.is_wild()]
        if wilds:
            chosen = wilds[0]
            color_counts = {c: sum(1 for card in hand if card.color == c) for c in COLORS}
            best_color = max(color_counts, key=color_counts.get) or "RED"
            return chosen, best_color

        return None


class UnoGameSimulator:
    """Simulates a 4-player UNO game to completion."""

    def __init__(self, strategies: List[PlayerStrategy]):
        assert len(strategies) == 4, "Simulation requires exactly 4 players"
        self.strategies = strategies

    def play_game(self) -> Dict[str, Any]:
        deck = build_deck()
        discard_pile: List[Card] = []

        # Deal 7 cards to each of the 4 players
        hands: List[List[Card]] = [[] for _ in range(4)]
        for _ in range(7):
            for p in range(4):
                hands[p].append(deck.pop())

        # Initial card from deck (must not be wild for simplicity)
        initial = deck.pop()
        while initial.is_wild():
            deck.insert(0, initial)
            initial = deck.pop()

        discard_pile.append(initial)
        active_color = initial.color
        active_value = initial.value

        current_player = 0
        direction = 1  # 1 for clockwise, -1 for counter-clockwise
        turns = 0
        cards_drawn = 0
        actions_triggered = 0

        max_turns = 1000  # Avoid infinite loops on abnormal deck exhaustion
        winner: Optional[int] = None

        while turns < max_turns:
            turns += 1
            hand = hands[current_player]
            strat = self.strategies[current_player]

            move = strat.choose_card(hand, active_color, active_value)

            if move is None:
                # Draw card
                if not deck:
                    if len(discard_pile) > 1:
                        top = discard_pile.pop()
                        deck = discard_pile[:]
                        discard_pile = [top]
                        random.shuffle(deck)
                    else:
                        break

                drawn = deck.pop() if deck else None
                if drawn:
                    hand.append(drawn)
                    cards_drawn += 1
                    # Immediate play check if drawn card is valid
                    if drawn.can_play_on(active_color, active_value):
                        move_drawn = strat.choose_card(hand, active_color, active_value)
                        if move_drawn and move_drawn[0] == drawn:
                            move = move_drawn
            if move is not None:
                card_to_play, declared_color = move
                hand.remove(card_to_play)
                discard_pile.append(card_to_play)
                active_color = declared_color
                active_value = card_to_play.value

                # Check win condition
                if len(hand) == 0:
                    winner = current_player
                    break

                # Handle action cards
                if card_to_play.value == "REVERSE":
                    direction *= -1
                    actions_triggered += 1
                elif card_to_play.value == "SKIP":
                    current_player = (current_player + direction) % 4
                    actions_triggered += 1
                elif card_to_play.value == "DRAW_TWO":
                    next_p = (current_player + direction) % 4
                    for _ in range(2):
                        if deck:
                            hands[next_p].append(deck.pop())
                            cards_drawn += 1
                    current_player = (current_player + direction) % 4
                    actions_triggered += 1
                elif card_to_play.value == "WILD_DRAW_FOUR":
                    next_p = (current_player + direction) % 4
                    for _ in range(4):
                        if deck:
                            hands[next_p].append(deck.pop())
                            cards_drawn += 1
                    current_player = (current_player + direction) % 4
                    actions_triggered += 1

            current_player = (current_player + direction) % 4

        if winner is None:
            # Lowest card count wins on timeout
            winner = min(range(4), key=lambda p: len(hands[p]))

        return {
            "winner": winner,
            "turns": turns,
            "cards_drawn": cards_drawn,
            "actions_triggered": actions_triggered,
        }


def run_tournament(num_games: int = 1000) -> Dict[str, Any]:
    """Runs a 1000-game tournament and returns aggregated statistics."""
    strategies = [
        ColorMatchPriorityStrategy("Player 0 (Strategy: ColorMatchPriority)"),
        RandomStrategy("Player 1 (Strategy: Random)"),
        RandomStrategy("Player 2 (Strategy: Random)"),
        RandomStrategy("Player 3 (Strategy: Random)"),
    ]

    sim = UnoGameSimulator(strategies)
    wins = [0, 0, 0, 0]
    total_turns = 0
    total_drawn = 0

    for _ in range(num_games):
        res = sim.play_game()
        wins[res["winner"]] += 1
        total_turns += res["turns"]
        total_drawn += res["cards_drawn"]

    stats = {
        "num_games": num_games,
        "wins": wins,
        "win_rates": [w / num_games * 100.0 for w in wins],
        "avg_turns": total_turns / num_games,
        "avg_drawn": total_drawn / num_games,
        "strategies": [s.name for s in strategies],
    }
    return stats


def print_stats(stats: Dict[str, Any]):
    print("=" * 65)
    print(f"  UNO 4-PLAYER TOURNAMENT SIMULATION ({stats['num_games']:,} GAMES)")
    print("=" * 65)
    for p in range(4):
        print(f"  Player {p} ({stats['strategies'][p]}): {stats['wins'][p]:>4} wins ({stats['win_rates'][p]:>5.1f}%)")
    print("-" * 65)
    print(f"  Average Turns per Game : {stats['avg_turns']:.1f}")
    print(f"  Average Cards Drawn    : {stats['avg_drawn']:.1f}")
    print("=" * 65)


if __name__ == "__main__":
    results = run_tournament(1000)
    print_stats(results)
