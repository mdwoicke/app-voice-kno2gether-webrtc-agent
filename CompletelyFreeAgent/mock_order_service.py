import re
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

class MockSalonService:
    def __init__(self):
        self.bookings = []
        self.booking_counter = 1000
        self.business_hours = {
            'Monday': ('9:00', '20:00'),
            'Tuesday': ('9:00', '20:00'),
            'Wednesday': ('9:00', '20:00'),
            'Thursday': ('9:00', '20:00'),
            'Friday': ('9:00', '20:00'),
            'Saturday': ('9:00', '20:00'),
            'Sunday': ('10:00', '18:00')
        }

    def validate_phone_number(self, phone: str) -> bool:
        # Basic UK phone number validation
        uk_phone_pattern = r'^\d+$'
        return bool(re.match(uk_phone_pattern, phone))

    def validate_datetime(self, date_str: str, time_str: str) -> tuple[bool, str]:
        try:
            # Parse the date and time
            booking_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            
            # Check if it's in the past
            if booking_datetime < datetime.now():
                return False, "Cannot book appointments in the past"
            
            # Get day of week
            day_of_week = booking_datetime.strftime('%A')
            
            # Check if within business hours
            if day_of_week in self.business_hours:
                open_time = datetime.strptime(self.business_hours[day_of_week][0], "%H:%M").time()
                close_time = datetime.strptime(self.business_hours[day_of_week][1], "%H:%M").time()
                booking_time = booking_datetime.time()
                
                if not (open_time <= booking_time <= close_time):
                    return False, f"We're not open at {time_str} on {day_of_week}s"
            
            return True, "Valid date and time"
        except ValueError:
            return False, "Invalid date or time format"

    def process_order(self, booking_data: Dict) -> Dict:
        # Validate phone number
        if not self.validate_phone_number(booking_data.get('phone', '')):
            return {
                'success': False,
                'error': 'Invalid UK phone number format'
            }

        # Validate date and time if provided
        date_str = booking_data.get('preferred_date')
        time_str = booking_data.get('preferred_time')
        if date_str and time_str:
            is_valid, message = self.validate_datetime(date_str, time_str)
            if not is_valid:
                return {
                    'success': False,
                    'error': message
                }

        # Generate booking details
        booking_id = f"BKG{self.booking_counter}"
        self.booking_counter += 1
        
        booking = {
            'booking_id': booking_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'confirmed',
            **booking_data
        }
        
        self.bookings.append(booking)
        
        return {
            'success': True,
            'booking_id': booking_id,
            'message': 'Booking confirmed'
        }

    def get_booking_status(self, booking_id: str) -> Optional[Dict]:
        for booking in self.bookings:
            if booking['booking_id'] == booking_id:
                return booking
        return None

# Global instance
salon_service = MockSalonService()
