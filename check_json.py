import json
from typing import Dict, List, Union, Optional
from datetime import datetime
import logging
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricingSchemaValidator:
    """Validator for pricing data schema"""
    
    def __init__(self):
        self.schema = {
            "type": "object",
            "required": ["service", "pricing_metadata", "plans", "discounts", "enterprise", "ml_metadata", "agent_metadata"],
            "properties": {
                "service": {
                    "type": "object",
                    "required": ["name", "url", "logo_url", "description", "category", "business_model"],
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "url": {"type": ["string", "null"]},
                        "logo_url": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "category": {
                            "type": "object",
                            "required": ["primary", "secondary", "tags"],
                            "properties": {
                                "primary": {"type": ["string", "null"]},
                                "secondary": {"type": ["string", "null"]},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "business_model": {
                            "type": "object",
                            "required": ["type", "description"],
                            "properties": {
                                "type": {"type": "array", "items": {"type": "string"}},
                                "description": {"type": ["string", "null"]}
                            }
                        }
                    }
                },
                "pricing_metadata": {
                    "type": "object",
                    "required": ["last_updated", "currency", "regions", "billing_cycles", "custom_pricing_available", "free_tier_available", "versioning"],
                    "properties": {
                        "last_updated": {"type": ["string", "null"]},
                        "currency": {"type": ["string", "null"]},
                        "regions": {"type": "array", "items": {"type": "string"}},
                        "billing_cycles": {
                            "type": "object",
                            "required": ["available", "default"],
                            "properties": {
                                "available": {"type": "array", "items": {"type": "string"}},
                                "default": {"type": ["string", "null"]}
                            }
                        },
                        "custom_pricing_available": {"type": ["boolean", "null"]},
                        "free_tier_available": {"type": ["boolean", "null"]},
                        "versioning": {
                            "type": "object",
                            "required": ["current", "history"],
                            "properties": {
                                "current": {"type": ["string", "null"]},
                                "history": {"type": "array"}
                            }
                        }
                    }
                },
                "plans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "slug", "description", "highlight_features", "is_popular", "pricing", "limits", "features"],
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "slug": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]},
                            "highlight_features": {"type": "array", "items": {"type": "string"}},
                            "is_popular": {"type": ["boolean", "null"]},
                            "pricing": {
                                "type": "object",
                                "required": ["base", "usage_based"],
                                "properties": {
                                    "base": {
                                        "type": "object",
                                        "required": ["amount", "amount_range", "period", "currency", "is_per_user"],
                                        "properties": {
                                            "amount": {"type": ["number", "null"]},
                                            "amount_range": {
                                                "type": "object",
                                                "required": ["min", "max"],
                                                "properties": {
                                                    "min": {"type": ["number", "null"]},
                                                    "max": {"type": ["number", "null"]}
                                                }
                                            },
                                            "period": {"type": ["string", "null"]},
                                            "currency": {"type": ["string", "null"]},
                                            "is_per_user": {"type": ["boolean", "null"]}
                                        }
                                    },
                                    "usage_based": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["name", "type", "unit", "tiers"],
                                            "properties": {
                                                "name": {"type": ["string", "null"]},
                                                "type": {"type": ["string", "null"]},
                                                "unit": {"type": ["string", "null"]},
                                                "tiers": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "required": ["range", "unit_price", "flat_fee"],
                                                        "properties": {
                                                            "range": {
                                                                "type": "object",
                                                                "required": ["min", "max"],
                                                                "properties": {
                                                                    "min": {"type": ["number", "null"]},
                                                                    "max": {"type": ["number", "null"]}
                                                                }
                                                            },
                                                            "unit_price": {"type": ["number", "null"]},
                                                            "flat_fee": {"type": ["number", "null"]}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "discounts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "amount", "description", "conditions", "valid_until"],
                        "properties": {
                            "type": {"type": ["string", "null"]},
                            "amount": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]},
                            "conditions": {"type": ["string", "null"]},
                            "valid_until": {"type": ["string", "null"]}
                        }
                    }
                },
                "enterprise": {
                    "type": "object",
                    "required": ["available", "contact_sales", "minimum_seats", "custom_features"],
                    "properties": {
                        "available": {"type": ["boolean", "null"]},
                        "contact_sales": {"type": ["boolean", "null"]},
                        "minimum_seats": {"type": ["number", "null"]},
                        "custom_features": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "ml_metadata": {
                    "type": "object",
                    "required": ["embeddings", "confidence_scores", "last_validated"],
                    "properties": {
                        "embeddings": {
                            "type": "object",
                            "required": ["model", "version", "vectors"],
                            "properties": {
                                "model": {"type": ["string", "null"]},
                                "version": {"type": ["string", "null"]},
                                "vectors": {"type": "array"}
                            }
                        },
                        "confidence_scores": {
                            "type": "object",
                            "required": ["pricing_accuracy", "feature_accuracy"],
                            "properties": {
                                "pricing_accuracy": {"type": ["number", "null"]},
                                "feature_accuracy": {"type": ["number", "null"]}
                            }
                        },
                        "last_validated": {"type": ["string", "null"]}
                    }
                },
                "agent_metadata": {
                    "type": "object",
                    "required": ["agent_name", "agent_version", "execution_time_seconds", "fallback_used", "comments"],
                    "properties": {
                        "agent_name": {"type": ["string", "null"]},
                        "agent_version": {"type": ["string", "null"]},
                        "execution_time_seconds": {"type": ["number", "null"]},
                        "fallback_used": {"type": ["boolean", "null"]},
                        "comments": {"type": ["string", "null"]}
                    }
                }
            }
        }

    def validate_pricing_data(self, data: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate pricing data against schema
        Returns (is_valid, error_message)
        """
        try:
            validate(instance=data, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, f"Validation error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def validate_file(self, file_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
        """
        Validate JSON file against schema
        Returns (is_valid, error_message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.validate_pricing_data(data)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON file: {str(e)}"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

def main():
    """Example usage of the schema validator"""
    validator = PricingSchemaValidator()
    
    # Test with a sample file
    test_file = Path("parsed_content_storage/parsed_airtable_20241217_021126.json")
    if test_file.exists():
        is_valid, error = validator.validate_file(test_file)
        if is_valid:
            logger.info(f"✅ {test_file.name} is valid")
        else:
            logger.error(f"❌ {test_file.name} is invalid: {error}")
    else:
        logger.warning(f"Test file not found: {test_file}")

if __name__ == "__main__":
    main()
