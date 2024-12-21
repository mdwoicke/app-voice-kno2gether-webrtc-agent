import re
import json
from datetime import datetime
from typing import Dict, Optional

class MockOrderService:
    def __init__(self):
        self.orders = []
        self.order_counter = 1000

    def validate_uk_postcode(self, postcode: str) -> bool:
        # Basic UK postcode validation
        uk_postcode_pattern = r'^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$'
        return bool(re.match(uk_postcode_pattern, postcode.upper()))

    def validate_phone_number(self, phone: str) -> bool:
        # Basic UK phone number validation
        uk_phone_pattern = r'^(?:(?:\+44)|(?:0))(?:[0-9] ?){9,10}$'
        return bool(re.match(uk_phone_pattern, phone))

    def process_order(self, order_data: Dict) -> Dict:
        # Extract postcode from address
        address = order_data.get('address', '').upper()
        postcode_match = re.search(r'[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}', address)
        
        if not postcode_match:
            return {
                'success': False,
                'error': 'Invalid UK postcode format in address'
            }

        if not self.validate_phone_number(order_data.get('phone', '')):
            return {
                'success': False,
                'error': 'Invalid UK phone number format'
            }

        # Generate order details
        order_id = f"ORD{self.order_counter}"
        self.order_counter += 1
        
        order = {
            'order_id': order_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'confirmed',
            'estimated_delivery': '30-45 minutes',
            **order_data
        }
        
        self.orders.append(order)
        
        return {
            'success': True,
            'order_id': order_id,
            'message': f'Order {order_id} has been confirmed'
        }

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        for order in self.orders:
            if order['order_id'] == order_id:
                return order
        return None

# Global instance
order_service = MockOrderService()
