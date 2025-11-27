import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

logger = logging.getLogger(__name__)

# Import Config for timezone setting
try:
    from config import Config
    DEFAULT_TIMEZONE = Config.TIMEZONE
except ImportError:
    DEFAULT_TIMEZONE = "Australia/Perth"  # AWST (UTC+8)


class HumanScheduler:
    """
    Simulates human-like posting behavior with randomized scheduling.
    Considers time zones, posting patterns, and natural breaks.
    """

    def __init__(self, timezone: str = None):
        """
        Initialize scheduler with target timezone.
        Default to Perth (AWST) for Australian Western Standard Time.

        Args:
            timezone: Timezone string (e.g., "Australia/Perth"). Uses Config.TIMEZONE if not specified.
        """
        tz_name = timezone or DEFAULT_TIMEZONE
        self.timezone = pytz.timezone(tz_name)
        self.timezone_name = tz_name
        self.utc = pytz.UTC
        logger.info(f"Scheduler initialized with timezone: {tz_name}")

        # Define posting windows (local time)
        # Mimics when a human would naturally post
        self.posting_windows = {
            'morning': {'start': 7, 'end': 9, 'weight': 2},      # 7-9am: Coffee time scrolling
            'late_morning': {'start': 10, 'end': 12, 'weight': 3}, # 10am-12pm: Active work hours
            'lunch': {'start': 12, 'end': 14, 'weight': 2},       # 12-2pm: Lunch break posting
            'afternoon': {'start': 14, 'end': 17, 'weight': 3},   # 2-5pm: Afternoon productivity
            'evening': {'start': 18, 'end': 21, 'weight': 4},     # 6-9pm: Peak social media
            'night': {'start': 21, 'end': 24, 'weight': 2},       # 9pm-midnight: Late scrolling
        }

        # Days with different posting patterns
        self.day_weights = {
            0: 0.8,   # Monday - slower start
            1: 1.0,   # Tuesday - full engagement
            2: 1.0,   # Wednesday - full engagement
            3: 1.0,   # Thursday - full engagement
            4: 1.1,   # Friday - slightly more active
            5: 0.6,   # Saturday - weekend slowdown
            6: 0.5,   # Sunday - minimal posting
        }

        # Post type preferences by time
        self.time_preferences = {
            'morning': ['text', 'infographic'],  # Quick reads in morning
            'late_morning': ['video', 'image', 'infographic'],  # Visual content
            'lunch': ['meme', 'text'],  # Light content at lunch
            'afternoon': ['image', 'video', 'infographic'],  # Substantive content
            'evening': ['video', 'meme', 'image'],  # Engaging evening content
            'night': ['meme', 'text'],  # Quick late-night content
        }

    def get_current_window(self) -> Optional[str]:
        """
        Returns the current posting window name based on local time.
        """
        now = datetime.now(self.timezone)
        hour = now.hour

        for window_name, window in self.posting_windows.items():
            if window['start'] <= hour < window['end']:
                return window_name

        return None  # Outside posting hours

    def should_post_now(self) -> Tuple[bool, str]:
        """
        Determines if the bot should post right now.
        Returns (should_post: bool, reason: str).

        Uses higher probability to ensure consistent posting.
        """
        now = datetime.now(self.timezone)
        hour = now.hour
        day_of_week = now.weekday()

        # Check if within any posting window
        current_window = self.get_current_window()
        if not current_window:
            return False, f"Outside posting hours ({hour}:00 local)"

        # Apply day weight (weekends slightly less active)
        day_weight = self.day_weights.get(day_of_week, 1.0)
        window_weight = self.posting_windows[current_window]['weight']

        # Higher probability to ensure posts happen
        # Base 70% * day_weight * normalized window weight
        # Peak times (evening, afternoon) get ~80-90% probability
        # Off-peak times get ~50-60% probability
        normalized_weight = window_weight / 4  # max weight is 4
        post_probability = 0.70 * day_weight * (0.7 + 0.3 * normalized_weight)

        # Clamp between 40% and 95%
        post_probability = max(0.40, min(0.95, post_probability))

        # Random decision
        roll = random.random()
        should_post = roll < post_probability

        if should_post:
            return True, f"Posting during {current_window} window (prob: {post_probability:.2f})"
        else:
            return False, f"Skipping this trigger (roll: {roll:.2f} >= prob: {post_probability:.2f})"

    def get_preferred_post_types(self) -> List[str]:
        """
        Returns preferred post types for current time window.
        """
        current_window = self.get_current_window()
        if current_window and current_window in self.time_preferences:
            return self.time_preferences[current_window]
        return ['text', 'image', 'video', 'meme', 'infographic']  # Default: all types

    def get_posts_per_day_target(self) -> int:
        """
        Returns target number of posts for today based on day of week.
        Adds randomization for human-like variance.
        """
        now = datetime.now(self.timezone)
        day_of_week = now.weekday()
        day_weight = self.day_weights.get(day_of_week, 1.0)

        # Base: 3-5 posts per day, modified by day weight
        base_posts = random.randint(3, 5)
        adjusted = int(base_posts * day_weight)

        return max(1, adjusted)  # At least 1 post per day

    def get_next_post_delay(self) -> timedelta:
        """
        Returns a randomized delay until next post.
        Used when running continuous scheduler.
        """
        current_window = self.get_current_window()

        if not current_window:
            # Outside posting hours, wait until next window
            now = datetime.now(self.timezone)
            for window_name, window in self.posting_windows.items():
                if window['start'] > now.hour:
                    hours_until = window['start'] - now.hour
                    return timedelta(hours=hours_until, minutes=random.randint(0, 30))

            # Next window is tomorrow morning
            hours_until_morning = (24 - now.hour) + self.posting_windows['morning']['start']
            return timedelta(hours=hours_until_morning, minutes=random.randint(0, 30))

        # Within posting hours, space posts naturally
        # Human-like: 2-4 hours between posts with some variance
        base_minutes = random.randint(90, 240)  # 1.5 to 4 hours
        variance_minutes = random.randint(-30, 30)  # +/- 30 min variance

        return timedelta(minutes=max(60, base_minutes + variance_minutes))

    def generate_daily_schedule(self) -> List[Dict]:
        """
        Generates a randomized posting schedule for the day.
        Returns list of {time: datetime, preferred_types: List[str]}.
        """
        now = datetime.now(self.timezone)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        target_posts = self.get_posts_per_day_target()
        schedule = []

        # Distribute posts across active windows
        active_windows = []
        for window_name, window in self.posting_windows.items():
            # Weight affects how many posts land in this window
            active_windows.extend([window_name] * window['weight'])

        # Shuffle and pick
        random.shuffle(active_windows)
        selected_windows = active_windows[:target_posts]

        for window_name in selected_windows:
            window = self.posting_windows[window_name]

            # Random time within window
            hour = random.randint(window['start'], window['end'] - 1)
            minute = random.randint(0, 59)

            post_time = today.replace(hour=hour, minute=minute)

            # Skip if time has passed
            if post_time > now:
                schedule.append({
                    'time': post_time,
                    'window': window_name,
                    'preferred_types': self.time_preferences.get(window_name, ['text'])
                })

        # Sort by time
        schedule.sort(key=lambda x: x['time'])

        logger.info(f"Generated schedule for today: {len(schedule)} posts")
        for item in schedule:
            logger.info(f"  - {item['time'].strftime('%H:%M')} ({item['window']}): {item['preferred_types']}")

        return schedule

    def get_cron_expressions(self) -> List[str]:
        """
        Returns Cloud Scheduler cron expressions for multiple daily triggers.
        Creates 6-8 trigger points spread throughout the day.
        """
        cron_expressions = []

        # Generate trigger points for each window
        for window_name, window in self.posting_windows.items():
            # 1-2 triggers per window based on weight
            num_triggers = min(2, window['weight'])

            for i in range(num_triggers):
                # Spread triggers across the window
                window_span = window['end'] - window['start']
                hour = window['start'] + (i * window_span // num_triggers)

                # Random minute for each trigger
                minute = random.randint(0, 59)

                # Cron format: minute hour * * *
                cron_expressions.append(f"{minute} {hour} * * *")

        logger.info(f"Generated {len(cron_expressions)} cron triggers")
        return cron_expressions


def get_scheduler_config() -> Dict:
    """
    Returns scheduler configuration for deployment.
    """
    scheduler = HumanScheduler()

    return {
        'timezone': scheduler.timezone_name,
        'cron_expressions': scheduler.get_cron_expressions(),
        'daily_target': scheduler.get_posts_per_day_target(),
        'windows': scheduler.posting_windows
    }


def should_post_lightweight() -> Tuple[bool, str]:
    """
    Lightweight scheduler check - can be called before initializing heavy Brain.
    This is a standalone function for cold start optimization.

    Returns:
        Tuple[bool, str]: (should_post, reason)
    """
    scheduler = HumanScheduler()
    return scheduler.should_post_now()
